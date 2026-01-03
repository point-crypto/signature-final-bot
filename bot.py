import os, cv2
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from fpdf import FPDF

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, ContextTypes, filters
)

# ================= CONFIG =================
import os
TOKEN = os.getenv("BOT_TOKEN")

REFERENCE, WAIT_TEST = range(2)

REF_DIR = "data/refs"
VIS_DIR = "visuals"
REPORT_DIR = "reports"

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ================= PDF SAFE =================
def pdf_safe(text):
    return (
        str(text)
        .replace("✔","").replace("❌","").replace("✅","")
        .replace("🔥","").replace("📊","").replace("➡","->")
        .replace("•","-").replace("🟢","").replace("🟡","").replace("🔴","")
    )

# ================= PREPROCESS =================
def preprocess(path, canvas=(300,150)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, bin_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    x,y,w,h = cv2.boundingRect(np.vstack(cnts))
    ink = bin_img[y:y+h, x:x+w]

    scale = min(canvas[0]/w, canvas[1]/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(ink,(nw,nh))

    canvas_img = np.zeros((canvas[1],canvas[0]),dtype=np.uint8)
    xo, yo = (canvas[0]-nw)//2, (canvas[1]-nh)//2
    canvas_img[yo:yo+nh, xo:xo+nw] = resized
    return canvas_img

# ================= BLANK CHECK =================
def is_blank_signature(img, min_ink_ratio=0.003):
    if img is None:
        return True
    ink = np.sum(img > 0)
    return (ink / img.size) < min_ink_ratio

# ================= PEN PRESSURE =================
def pen_pressure(img):
    ink = img[img > 0]
    return float(np.mean(ink)) if len(ink) else 0.0

# ================= FEATURES =================
def stroke_density(img):
    return np.sum(img>0)/img.size

def contour_count(img):
    e = cv2.Canny(img,50,150)
    c,_ = cv2.findContours(e,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return len(c)

def hu_similarity(a,b):
    ha = cv2.HuMoments(cv2.moments(a)).flatten()
    hb = cv2.HuMoments(cv2.moments(b)).flatten()
    ha = -np.sign(ha)*np.log10(np.abs(ha)+1e-10)
    hb = -np.sign(hb)*np.log10(np.abs(hb)+1e-10)
    return max(0, 100 - np.mean(np.abs(ha-hb))*20)

def hist_similarity(a,b):
    ha = cv2.calcHist([a],[0],None,[256],[0,256])
    hb = cv2.calcHist([b],[0],None,[256],[0,256])
    ha, hb = cv2.normalize(ha,ha), cv2.normalize(hb,hb)
    return max(0, cv2.compareHist(ha,hb,cv2.HISTCMP_CORREL)*100)

def sim(a,b):
    return max(0, 100 - abs(a-b)*100)

# ================= DYNAMIC THRESHOLD =================
def dynamic_threshold(refs):
    if len(refs) < 2:
        return 65.0
    sims=[]
    for i in range(len(refs)):
        for j in range(i+1,len(refs)):
            sims.append(ssim(refs[i],refs[j])*100)
    return float(min(max(np.mean(sims)-(5+np.std(sims)),60),80))

# ================= VISUALS =================
def confidence_graph(score):
    p=f"{VIS_DIR}/confidence.png"
    plt.figure()
    plt.bar(["Confidence","Forgery Risk"],[score,100-score])
    plt.ylim(0,100)
    plt.title("Overall Verification Confidence")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(p); plt.close()
    return p

def og_comparison_bar(scores):
    p=f"{VIS_DIR}/og_comparison.png"
    plt.figure(figsize=(6,4))
    plt.bar([f"OG-{i+1}" for i in range(len(scores))],scores)
    plt.ylim(0,100)
    plt.title("Original vs Questioned Signature Similarity")
    plt.xlabel("Reference Signatures")
    plt.ylabel("Similarity (%)")
    plt.grid(axis="y",alpha=0.4)
    plt.tight_layout()
    plt.savefig(p); plt.close()
    return p

def majority_vote_pie(m,n):
    p=f"{VIS_DIR}/majority_vote.png"
    plt.figure()
    plt.pie([m,n],labels=["Match","Mismatch"],autopct="%1.1f%%",startangle=90)
    plt.title("Majority Vote Decision")
    plt.tight_layout()
    plt.savefig(p); plt.close()
    return p

# ================= AI EXPLANATION =================
def ai_explanation(score,thr,m,t,pp_ref,pp_test):
    pressure_reason = (
        "• Pen pressure of the questioned signature is consistent with reference samples."
        if abs(pp_ref-pp_test) < 15
        else "• Noticeable pen pressure deviation detected compared to reference samples."
    )

    if score>=thr:
        return (
            f"• {m} out of {t} reference signatures support a MATCH decision.\n"
            "• Stroke structure, contours, and overall shape are consistent.\n"
            f"{pressure_reason}\n"
            "• Robust aggregation and majority voting confirm authenticity."
        )
    else:
        return (
            f"• Only {m} out of {t} reference signatures support a MATCH decision.\n"
            "• Significant deviations observed in stroke structure and shape.\n"
            f"{pressure_reason}\n"
            "• Majority disagreement leads to a FORGERY classification."
        )

# ================= PDF =================
def generate_pdf(report,imgs):
    pdf=FPDF()
    pdf.add_page()
    pdf.set_font("Arial",size=12)
    pdf.cell(0,10,"Signature Verification Report",ln=True)
    pdf.ln(5)
    for k,v in report.items():
        pdf.multi_cell(0,8,f"{k}: {pdf_safe(v)}")
    pdf.ln(5)
    for i in imgs:
        pdf.image(i,w=170); pdf.ln(4)
    p=f"{REPORT_DIR}/report_{int(datetime.now().timestamp())}.pdf"
    pdf.output(p)
    return p

# ================= BOT =================
async def start(update:Update,context:ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "✍️ *Signature Verification Bot*\n\n"
        "⚠️ *Important Warning*\n"
        "• Do NOT upload blank or faint images\n"
        "• Blank inputs will be rejected\n\n"
        "📌 *How to Use*\n"
        "1️⃣ Upload 2–4 ORIGINAL reference signatures\n"
        "2️⃣ Type /verify\n"
        "3️⃣ Upload ONE questioned signature\n\n"
        "🔍 *System Analysis*\n"
        "• Stroke & shape similarity\n"
        "• Pen pressure estimation\n"
        "• Majority vote decision",
        parse_mode="Markdown"
    )
    return REFERENCE

async def save_reference(update:Update,context:ContextTypes.DEFAULT_TYPE):
    uid=str(update.message.from_user.id)
    udir=os.path.join(REF_DIR,uid)
    os.makedirs(udir,exist_ok=True)
    f=await update.message.photo[-1].get_file()
    await f.download_to_drive(os.path.join(udir,f"ref_{len(os.listdir(udir))+1}.jpg"))
    await update.message.reply_text("✅ Reference saved. Send more or type /verify")
    return REFERENCE

async def verify(update:Update,context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📤 Upload ONE questioned signature")
    return WAIT_TEST

async def test_signature(update:Update,context:ContextTypes.DEFAULT_TYPE):
    uid=str(update.message.from_user.id)
    udir=os.path.join(REF_DIR,uid)

    f=await update.message.photo[-1].get_file()
    await f.download_to_drive("test.jpg")
    test=preprocess("test.jpg")

    if is_blank_signature(test):
        await update.message.reply_text(
            "❌ *Invalid Input*\n"
            "Blank or faint signatures reduce accuracy.\n"
            "Please upload a clear handwritten signature.",
            parse_mode="Markdown"
        )
        return ConversationHandler.END

    refs=[preprocess(os.path.join(udir,r)) for r in os.listdir(udir)]
    refs=[r for r in refs if r is not None]

    thr=dynamic_threshold(refs)
    scores=[]; m=0
    pp_test=pen_pressure(test); pp_refs=[]

    for r in refs:
        pp_refs.append(pen_pressure(r))
        s1=ssim(r,test)*100
        s2=sim(stroke_density(r),stroke_density(test))
        s3=sim(contour_count(r),contour_count(test))
        s4=hu_similarity(r,test)
        s5=hist_similarity(r,test)
        final=0.4*s1+0.15*s2+0.15*s3+0.2*s4+0.1*s5
        scores.append(final)
        if final>=thr: m+=1

    score=max(np.median(scores),np.mean(sorted(scores)[-max(2,int(len(scores)*0.7)):]))

    imgs=[
        confidence_graph(score),
        og_comparison_bar(scores),
        majority_vote_pie(m,len(scores)-m)
    ]

    explanation=ai_explanation(score,thr,m,len(scores),np.mean(pp_refs),pp_test)

    report={
        "Final Score":f"{score:.2f}%",
        "Threshold":f"{thr:.2f}%",
        "Pen Pressure (Reference Avg)":f"{np.mean(pp_refs):.2f}",
        "Pen Pressure (Test)":f"{pp_test:.2f}",
        "Result":"MATCH" if score>=thr else "MISMATCH",
        "AI Explanation":explanation
    }

    generate_pdf(report,imgs)

    await update.message.reply_text(
        f"🔍 *Result*\n\n"
        f"Score: `{score:.2f}%`\n"
        f"Threshold: `{thr:.2f}%`\n"
        f"Pen Pressure (Ref/Test): `{np.mean(pp_refs):.1f} / {pp_test:.1f}`\n"
        f"*{report['Result']}*\n\n"
        f"*AI Result Reasons*\n{explanation}",
        parse_mode="Markdown"
    )

    captions=[
        "📊 Overall Verification Confidence",
        "📈 OG vs Questioned Signature Comparison",
        "🗳 Majority Vote Decision"
    ]

    for img,cap in zip(imgs,captions):
        await update.message.reply_photo(open(img,"rb"),caption=cap)

    for f in os.listdir(udir):
        try: os.remove(os.path.join(udir,f))
        except: pass

    return ConversationHandler.END

# ================= MAIN =================
def main():
    app=Application.builder().token(TOKEN).build()
    conv=ConversationHandler(
        entry_points=[CommandHandler("start",start)],
        states={
            REFERENCE:[MessageHandler(filters.PHOTO,save_reference),
                       CommandHandler("verify",verify)],
            WAIT_TEST:[MessageHandler(filters.PHOTO,test_signature)]
        },
        fallbacks=[CommandHandler("start",start)]
    )
    app.add_handler(conv)
    print("🤖 Bot running with titles & bullet-point explanations")
    app.run_polling()

if __name__=="__main__":
    main()

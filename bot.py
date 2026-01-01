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
TOKEN = "PASTE_YOUR_BOT_TOKEN_HERE"

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

    mean_sim = np.mean(sims)
    std_sim = np.std(sims)

    threshold = mean_sim - (5 + std_sim)
    return float(min(max(threshold,60),80))

# ================= VISUALS =================
def heatmap(a,b):
    d=cv2.absdiff(a,b)
    h=cv2.applyColorMap(d,cv2.COLORMAP_JET)
    p=f"{VIS_DIR}/heatmap.png"
    cv2.imwrite(p,h)
    return p

def confidence_graph(score):
    p=f"{VIS_DIR}/confidence.png"
    plt.figure()
    plt.bar(["Confidence","Forgery"],[score,100-score])
    plt.ylim(0,100)
    plt.savefig(p); plt.close()
    return p

# ================= PDF =================
def generate_pdf(report, imgs):
    pdf=FPDF()
    pdf.add_page()
    pdf.set_font("Arial",size=12)

    pdf.cell(0,10,"Signature Verification Report",ln=True)
    pdf.ln(5)

    for k,v in report.items():
        pdf.multi_cell(0,8,f"{k}: {pdf_safe(v)}")

    pdf.ln(5)
    for i in imgs:
        pdf.image(i,w=170)
        pdf.ln(5)

    path=f"{REPORT_DIR}/report_{int(datetime.now().timestamp())}.pdf"
    pdf.output(path)
    return path

# ================= BOT =================
async def start(update:Update, context:ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "✍️ Signature Verification Bot\n\n"
        "1️⃣ Upload 2–4 ORIGINAL signatures\n"
        "2️⃣ Type /verify\n"
        "3️⃣ Upload ONE questioned signature\n\n"
        "ℹ️ References are used for one verification only."
    )
    return REFERENCE

async def save_reference(update:Update, context:ContextTypes.DEFAULT_TYPE):
    uid=str(update.message.from_user.id)
    udir=os.path.join(REF_DIR,uid)
    os.makedirs(udir,exist_ok=True)

    f=await update.message.photo[-1].get_file()
    path=os.path.join(udir,f"ref_{len(os.listdir(udir))+1}.jpg")
    await f.download_to_drive(path)

    await update.message.reply_text("✅ Reference saved. Send more or type /verify")
    return REFERENCE

async def verify(update:Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📤 Upload ONE questioned signature")
    return WAIT_TEST

async def test_signature(update:Update, context:ContextTypes.DEFAULT_TYPE):
    uid=str(update.message.from_user.id)
    udir=os.path.join(REF_DIR,uid)

    f=await update.message.photo[-1].get_file()
    test_path="test.jpg"
    await f.download_to_drive(test_path)
    test=preprocess(test_path)

    refs=[preprocess(os.path.join(udir,r)) for r in os.listdir(udir)]
    refs=[r for r in refs if r is not None]

    thr=dynamic_threshold(refs)

    scores=[]
    best_ref=None
    best_ssim=0

    for r in refs:
        s1=ssim(r,test)*100
        s2=sim(stroke_density(r),stroke_density(test))
        s3=sim(contour_count(r),contour_count(test))
        s4=hu_similarity(r,test)
        s5=hist_similarity(r,test)

        final=(
            0.4*s1 +
            0.15*s2 +
            0.15*s3 +
            0.2*s4 +
            0.1*s5
        )
        scores.append(final)

        if s1>best_ssim:
            best_ssim=s1
            best_ref=r

    # ---------- ROBUST AGGREGATION ----------
    scores_sorted=sorted(scores)
    median_score=float(np.median(scores_sorted))
    k=max(2,int(len(scores_sorted)*0.7))
    best_k_avg=float(np.mean(scores_sorted[-k:]))
    score=max(median_score,best_k_avg)

    risk=100-score
    result="MATCH" if score>=thr else "MISMATCH"

    h=heatmap(best_ref,test)
    g=confidence_graph(score)

    report={
        "Score":f"{score:.2f}%",
        "Threshold":f"{thr:.2f}%",
        "Result":result,
        "Forgery Risk":f"{risk:.2f}%"
    }

    generate_pdf(report,[h,g])

    await update.message.reply_text(
        f"🔍 Result\n\n"
        f"Score: {score:.2f}%\n"
        f"Threshold: {thr:.2f}%\n"
        f"{result}\n"
        f"Forgery Risk: {risk:.2f}%"
    )

    await update.message.reply_photo(open(h,"rb"),caption="🔥 Heatmap")
    await update.message.reply_photo(open(g,"rb"),caption="📊 Confidence Graph")

    # Clear references safely (Windows-safe)
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
    print("🤖 Bot running perfectly")
    app.run_polling()

if __name__=="__main__":
    main()

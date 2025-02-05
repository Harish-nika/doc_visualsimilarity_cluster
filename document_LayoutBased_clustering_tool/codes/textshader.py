import pypdfium2 as pdfium
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
import sqlite3, os, sys
from tqdm import tqdm

path = "/FERack11_FE_documents2/EMMA_Official_Statement"   # Path to the server folder containing the PDFs
sys.path.append(path)

def numrows(db, query):
    db.execute(query)
    rows = db.fetchall()
    return len(rows)

def convert_image_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def threshold_image(grayscale_image):
    return cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
def invert_image(thresholded_image):
    return cv2.bitwise_not(thresholded_image)
def dilate_image(inverted_image, iterations=4):
    return cv2.dilate(inverted_image, None, iterations=iterations)

def erode_image(inverted_image, iterations=4):
    return cv2.erode(inverted_image, None, iterations=iterations)

def analyze_doc_skeleton(pdf_page, scale):
    page_image = pdf_page.render(scale=scale).to_pil()
    image = np.array(page_image)
    gs_img = convert_image_to_grayscale(image)
    th_img = threshold_image(gs_img)
    inv_img = invert_image(th_img)
    dilated_image = dilate_image(inv_img, iterations=13)   #text shading value
    img = Image.fromarray(dilated_image)
    return img

def analyze_pdf(pdf_paths, output_image_path):
    for file_path in tqdm(pdf_paths, desc="Processing PDFs", unit="file"):
        pdf = pdfium.PdfDocument(os.path.join(path, f"{file_path}.pdf"))
        page_no = 0
        pdf_name = Path(file_path).stem
        pdf_page = pdf[page_no]
        skeleton = analyze_doc_skeleton(pdf_page, scale=4)
        skeleton.save(f"{output_image_path}/{pdf_name}-{page_no}.png")
        
        with sqlite3.connect("/home/harish/workspace_dc/document_LayoutBased_clustering_tool/Database/document_clustering_sample.db", timeout=120) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE doc_clustering_sample
                    SET status = ?
                    WHERE filename = ?
                ''', ('PROCESSED', file_path))
                conn.commit()

if __name__ == "__main__":
    pdf_paths = []
    
    with sqlite3.connect("/home/harish/workspace_dc/document_LayoutBased_clustering_tool/Database/document_clustering_sample.db", timeout=60) as conn:
        db = conn.cursor()
        records = numrows(db, query = "SELECT * FROM doc_clustering_sample where status = ''")
        
        if records != 0:
            db.execute('''
            SELECT * FROM doc_clustering_sample where status = ''
            ''')
            rows = db.fetchall()
            pdf_paths = [row[0] for row in rows]
        else:
            print("*** Entire Process Completed ***")
            sys.exit(0)
    
    output_image_path = "/home/harish/workspace_dc/document_LayoutBased_clustering_tool/data_images" # zip -r doc_clustering_5K.zip final_out_set_2 (Command for zipping the folder)
    analyze_pdf(pdf_paths[:50], output_image_path)  # For testing purpose, only 50 PDFs are processed
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, Frame, Label, filedialog, Button, Toplevel, Canvas, Scrollbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

image_size = (128, 128)

def verify_single_image(model_path, image_path):
    try:
        model = load_model(model_path)
    except Exception as e:
        show_result_window(f"Error loading model: {e}")
        return
    if not os.path.exists(image_path):
        show_result_window("Image not found.")
        return
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            show_result_window("Invalid image file.")
            return
        img = cv2.resize(img, image_size) / 255.0
        img = img.reshape(1, image_size[0], image_size[1], 1)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        result = "Genuine" if predicted_class == 0 else "Forged"
        show_result_window(f"Result: {result} ({confidence:.2f}%)", predicted_class)
    except Exception as e:
        show_result_window(f"Error processing image: {e}")

def show_result_window(message, predicted_class=None):
    result_window = Toplevel()
    result_window.title("Prediction Result")
    result_window.geometry("900x600")
    result_window.configure(bg="#98FB98")
    Label(result_window, text=message, font=("Arial", 14, "bold"), bg="#98FB98", fg="#003366", pady=20).pack()
    if predicted_class is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ["Genuine", "Forged"]
        counts = [1, 0] if predicted_class == 0 else [0, 1]
        explode = (0.05, 0.05)
        ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=140, colors=["#66b3ff", "#ff9999"], shadow=True, explode=explode)
        ax.set_title("Verification Result")
        canvas_plot = FigureCanvasTkAgg(fig, master=result_window)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack(pady=20)
    Button(result_window, text="Close", command=result_window.destroy, font=("Arial", 12, "bold"), bg="#003366", fg="white").pack(pady=10)

def verify_signature_bundle(model_path, test_directory):
    try:
        model = load_model(model_path)
    except Exception as e:
        show_result_window(f"Error loading model: {e}")
        return
    if not os.path.exists(test_directory):
        show_result_window(f"The test directory {test_directory} does not exist.")
        return
    test_images = []
    filenames = []
    predicted_labels = []
    confidences = []
    original_count = 0
    forged_count = 0
    for root, dirs, files in os.walk(test_directory):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, filename)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, image_size) / 255.0
                    img = img.reshape(1, image_size[0], image_size[1], 1)
                    test_images.append((img, img_path))
                    filenames.append(filename)
                except Exception as e:
                    continue
    if not test_images:
        show_result_window("No valid images found in the test directory.")
        return
    confidences = []
    for i, (img, img_path) in enumerate(test_images):
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        result = "Genuine" if predicted_class == 0 else "Forged"
        predicted_labels.append((filenames[i], result, confidence, img_path))
        confidences.append(confidence)
        if predicted_class == 0:
            original_count += 1
        else:
            forged_count += 1
    display_results_window(predicted_labels, original_count, forged_count, confidences)

def display_results_window(predicted_labels, original_count, forged_count, confidences):
    result_window = Toplevel()
    result_window.title("Signature Verification Results")
    result_window.geometry("1400x800")
    result_window.configure(bg="#98FB98")
    left_frame = Frame(result_window, bg="#98FB98", width=500)
    left_frame.pack(side="left", fill="both", expand=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ['Original', 'Forged']
    counts = [original_count, forged_count]
    explode = (0.05, 0.05)
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'], shadow=True)
    ax.set_title('Overall Signature Verification Results')
    canvas_pie = FigureCanvasTkAgg(fig, master=left_frame)
    canvas_pie.draw()
    canvas_pie.get_tk_widget().pack(pady=10)
    fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
    ax_hist.hist(confidences, bins=10, color="#66b3ff", edgecolor="black")
    ax_hist.set_title("Confidence Distribution")
    ax_hist.set_xlabel("Confidence (%)")
    ax_hist.set_ylabel("Frequency")
    canvas_hist = FigureCanvasTkAgg(fig_hist, master=left_frame)
    canvas_hist.draw()
    canvas_hist.get_tk_widget().pack(pady=10)
    right_frame = Frame(result_window, bg="#98FB98", width=700)
    right_frame.pack(side="right", fill="both", expand=True)
    result_canvas = Canvas(right_frame, bg="#98FB98")
    scrollbar = Scrollbar(right_frame, orient="vertical", command=result_canvas.yview)
    result_frame = Frame(result_canvas, bg="#98FB98")
    result_frame.bind(
        "<Configure>",
        lambda e: result_canvas.configure(scrollregion=result_canvas.bbox("all"))
    )
    result_canvas.create_window((0, 0), window=result_frame, anchor="nw")
    result_canvas.configure(yscrollcommand=scrollbar.set)
    for filename, result, confidence, img_path in predicted_labels:
        color = '#008000' if result == 'Genuine' else '#FF0000'
        label_text = f"{filename}: {result} ({confidence:.2f}%)"
        lbl = Label(result_frame, text=label_text, fg=color, font=("Arial", 10), bg="#98FB98", pady=5)
        lbl.pack(anchor="e", padx=20)
        def open_image(path=img_path):
            img_window = Toplevel()
            img_window.title(filename)
            img = Image.open(path)
            img = img.resize((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            Label(img_window, image=img_tk).pack()
            img_window.mainloop()
        lbl.bind("<Button-1>", lambda e, path=img_path: open_image(path))
    scrollbar.pack(side="right", fill="y")
    result_canvas.pack(side="left", fill="both", expand=True)
    boxplot_frame = Frame(result_window, bg="#98FB98", width=200)
    boxplot_frame.pack(side="right", fill="y", expand=False)
    fig_box, ax_box = plt.subplots(figsize=(5, 4))
    ax_box.boxplot(confidences, vert=False, patch_artist=True, boxprops=dict(facecolor="#66b3ff"))
    ax_box.set_title("Confidence Boxplot")
    ax_box.set_xlabel("Confidence (%)")
    canvas_box = FigureCanvasTkAgg(fig_box, master=boxplot_frame)
    canvas_box.draw()
    canvas_box.get_tk_widget().pack(pady=10)
    Button(result_window, text="Close", command=result_window.destroy, font=("Arial", 12, "bold"), bg="#003366", fg="white").pack(pady=10)

def select_and_verify_dataset():
    dataset_dir = filedialog.askdirectory(title="Select the Dataset Folder")
    if dataset_dir:
        verify_signature_bundle(model_path, dataset_dir)

def select_and_verify_single_image():
    image_path = filedialog.askopenfilename(title="Select the Image for Verification", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        verify_single_image(model_path, image_path)

model_path = r"D:\signature_verification\signature_verification_model.keras"

main_window = Tk()
main_window.title("Signature Verification")
main_window.geometry("600x400")
main_window.configure(bg="#98FB98")
frame = Frame(main_window, bg="#98FB98")
frame.pack(expand=True, padx=20, pady=20)
Label(frame, text="Signature Verification Tool", font=("Arial", 18, "bold"), bg="#98FB98", fg="#003366").pack(pady=20)
Button(frame, text="Verify Single Image", command=select_and_verify_single_image, font=("Arial", 14, "bold"), bg="#003366", fg="white").pack(pady=10)
Button(frame, text="Verify Signature Dataset", command=select_and_verify_dataset, font=("Arial", 14, "bold"), bg="#003366", fg="white").pack(pady=10)
Button(frame, text="Exit", command=main_window.destroy, font=("Arial", 14, "bold"), bg="#003366", fg="white").pack(pady=10)
main_window.mainloop()

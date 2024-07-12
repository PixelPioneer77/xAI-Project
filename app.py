import ultralytics
import tkinter as tk
from tkinter import filedialog, Label, Button
from ultralytics import YOLO
import os
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


from ultralytics import YOLO
# base model
model_v8_base = YOLO('models/yolov8_base_weights.pt')
model_v8_base_path = 'models/yolov8_base_weights.pt'

#fine tuned
model_v8_fine = YOLO('models/yolov8_pre_trained.pt')
model_v8_fine_path = 'models/yolov8_pre_trained.pt'



# define what elements are forbidden (has to conform to the classes in the weapon fine tuned model)
forbidden_classes = ["guns", "knife"]


def detect_objects(model, image):
    results = model.predict(image,conf=0.2)
    detected_img = results[0].plot()
    #print(results[0])

    # Extract the predicted boxes
    boxes = results[0].boxes

    # save class names from prediction boxes
    class_names = []
    if boxes is not None:
        # Iterate through the detected boxes
        for box in boxes:
            cls_id = int(box.cls)  # Class ID
            conf = float(box.conf)  # Confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates

            # Get the class name from the class ID
            class_name = results[0].names[cls_id]
            class_names.append(class_name)

            print(f"Class: {class_name}, Confidence: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")



    # show with matplotlib
    #plt.imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))
    #plt.axis('off')
    #plt.show()

    return detected_img, class_names



from YOLOv8_Explainer import yolov8_heatmap, display_images

def generate_explanation(model_path, image_path):


  model_egc = yolov8_heatmap(
      weight= model_path,
      conf_threshold=0.4,
      method = "EigenGradCAM",
      layer=[10, 12, 14, 16, 18, -3],
      ratio=0.02,
      show_box=True,
      renormalize=False
  )

  model_pp = yolov8_heatmap(
      weight= model_path,
      conf_threshold=0.4,
      method = "GradCAMPlusPlus",
      layer=[10, 12, 14, 16, 18, -3],
      ratio=0.02,
      show_box=True,
      renormalize=False
  )

  model_lc = yolov8_heatmap(
      weight= model_path,
      conf_threshold=0.4,
      method = "LayerCAM",
      layer=[10, 12, 14, 16, 18, -3],
      ratio=0.02,
      show_box=True,
      renormalize=False
  )

  # set img path
  input_img_path = image_path


  # set img list
  imagelist = model_egc(img_path = image_path)

  return imagelist


def notify_uploader(image, explanation_image, message):
    # Send a notification with the message and explanation image
    # This can be done via email, messaging service, etc.
    # Placeholder function#
    message = "We are sorry to inform you that some content of your image does not comply with our terms and regulations. /n Our Detection Model detected forbidden items in your picture, attached is an explenation why your content is likely to be on our forbidden items list. If you feel this is a mistake you can contact us here..."
    print(message)
    #explanation_image.show()


def process_image(image_path, forbidden_classes, fine_tuned_model, original_model, fine_tuned_model_path, original_model_path):

    # Load image
    image = Image.open(image_path)
    #image_np = np.array(image)

    # Detect objects using the fine-tuned model
    fine_tuned_img, class_names = detect_objects(fine_tuned_model, image)
    # Check for forbidden objects
    is_any_item_forbidden = any(item in forbidden_classes for item in class_names)
    if is_any_item_forbidden == True:
        # Generate explanation using GradCam
                explanation_image = generate_explanation(fine_tuned_model_path, image_path)
                if explanation_image[0] == None:
                    message = f"Sorry, we were unable to generate an explenation for this Image. Please provide another picture."
                #print(explanation_image)


                # Notify uploader
                message = f"The image was restricted because it contains a forbidden object: {class_names}"
                notify_uploader(image, explanation_image, message)

                # Restrict the image (e.g., move to a restricted folder, delete, etc.)
                #restrict_image(image_path)

                return fine_tuned_img, explanation_image, message


                
    else:
        # Optionally, process with the original model for additional moderation
        results_original, class_names = detect_objects(original_model, image)
        if class_names is not None:
            explanation_image = generate_explanation(original_model_path, image_path)
            if explanation_image[0] == None:
                message = f"Sorry, we were unable to generate an explenation for this Image. Please provide another picture."
            else:
                message = f"Thank you your picture is good to go!"
        else:
            message = f"Sorry, we were unable to detect any objects in the image. Please provide another picture."
        
        # Further logic can be added here 
        #print(results_original, explanation_image)

    return results_original, explanation_image, message


import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import os
import cv2
from ultralytics import YOLO

# Function to upload and display the image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        original_img = Image.open(file_path)
        #original_img.thumbnail((400, 400))
        #original_img = ImageTk.PhotoImage(original_img)

        results_model, explanation_image, message = process_image(file_path,
                            forbidden_classes,
                            fine_tuned_model = model_v8_fine,
                            original_model = model_v8_base,
                            original_model_path = model_v8_base_path,
                            fine_tuned_model_path = model_v8_fine_path
                            )
    
        # Convert results_model to Image and then to ImageTk
        if isinstance(results_model, (list, tuple)):
            results_model = results_model[0]
        results_model_img = Image.fromarray(cv2.cvtColor(results_model, cv2.COLOR_BGR2RGB))
        results_model_img.thumbnail((800, 800))
        results_model_tk = ImageTk.PhotoImage(results_model_img)
        

        #process_image(file_path)
        print(explanation_image)
        if explanation_image[0] == None:
            processed_img = original_img
        else:
            processed_img = explanation_image[0]
        processed_img.thumbnail((800, 800))
        processed_img = ImageTk.PhotoImage(processed_img)

        panel_left.config(image=results_model_tk )
        panel_left.image = results_model_tk 
        panel_right.config(image=processed_img)
        panel_right.image = processed_img

        message_label.config(text=message)

# Function to clear the current image and reset the interface
def clear_file():
    panel_left.config(image='')
    panel_left.image = None
    panel_right.config(image='')
    panel_right.image = None
    message_label.config(text='')

# Create the main window
root = tk.Tk()
root.title("xAIM Weapon Content Moderation")
root.geometry("900x600")  # Fixed size window
root.configure(bg='#2E2E2E')  # Dark background for modern look

# Add logo and title
logo = Image.open("images/logo.jpg")  # Update with your logo path
logo.thumbnail((100, 100))
logo = ImageTk.PhotoImage(logo)
logo_label = Label(root, image=logo, bg='#2E2E2E')
logo_label.pack(pady=10)

title_label = Label(root, text="xAIM Weapon Content Moderation", font=("Helvetica", 24, "bold"), fg="#FFFFFF", bg='#2E2E2E')
title_label.pack(pady=10)

# Create and pack the widgets
frame = tk.Frame(root, bg='#2E2E2E')
frame.pack(pady=20)

panel_left = Label(frame, bg='#2E2E2E')
panel_left.grid(row=0, column=0, padx=20)

panel_right = Label(frame, bg='#2E2E2E')
panel_right.grid(row=0, column=1, padx=20)

message_label = Label(root, text="", font=("Helvetica", 14), fg="#FFFFFF", bg='#2E2E2E')
message_label.pack(pady=10)

btn_frame = tk.Frame(root, bg='#2E2E2E')
btn_frame.pack(pady=20)

btn_upload = Button(btn_frame, text="Upload an Image", command=upload_image, font=("Helvetica", 14), bg="#1E90FF", fg="#FFFFFF", activebackground="#104E8B", activeforeground="#FFFFFF")
btn_upload.grid(row=0, column=0, padx=10)

btn_clear = Button(btn_frame, text="Clear File", command=clear_file, font=("Helvetica", 14), bg="#FF6347", fg="#FFFFFF", activebackground="#CD5C5C", activeforeground="#FFFFFF")
btn_clear.grid(row=0, column=1, padx=10)

# Run the application
root.mainloop()

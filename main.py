import flet as ft

import cv2, os, csv, numpy as np

from PIL import Image

from openpyxl import Workbook, load_workbook

import datetime, time

import subprocess

import threading



# ---------------- CONFIG ----------------

BASE_DIR = r"C:\AI_ATTENDANCE"

STUDENT_DATA_DIR = os.path.join(BASE_DIR, "StudentData")

TRAINING_IMAGE_DIR = os.path.join(STUDENT_DATA_DIR, "TrainingImage")

TRAINING_IMAGE_LABEL_DIR = os.path.join(STUDENT_DATA_DIR, "TrainingImageLabel")

ATTENDANCE_FILE = os.path.join(STUDENT_DATA_DIR, "Attendance.xlsx")

LABEL_MAP_FILE = os.path.join(STUDENT_DATA_DIR, "LabelMap.csv")



xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

if not os.path.exists(xml_path):

    xml_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

CASCADE_PATH = xml_path



for folder in [STUDENT_DATA_DIR, TRAINING_IMAGE_DIR, TRAINING_IMAGE_LABEL_DIR]:

    os.makedirs(folder, exist_ok=True)



# ---------------- BACKEND HELPERS ----------------

def ensure_attendance_file(log_func):

    try:

        if not os.path.exists(ATTENDANCE_FILE):

            wb = Workbook()

            ws = wb.active

            ws.append(["ID", "Name", "Date", "Time"])

            wb.save(ATTENDANCE_FILE)

    except Exception as e:

        log_func(f"Error creating excel file: {e}")



def load_label_map(log_func):

    mapping = {}

    if os.path.exists(LABEL_MAP_FILE):

        try:

            with open(LABEL_MAP_FILE, "r", newline='', encoding='utf-8') as f:

                reader = csv.reader(f)

                for row in reader:

                    if len(row) >= 3:

                        mapping[int(row[0])] = (row[1], row[2])

        except Exception as e:

            log_func(f"Error reading label map: {e}")

    return mapping



def save_label_map(mapping):

    with open(LABEL_MAP_FILE, "w", newline='', encoding='utf-8') as f:

        writer = csv.writer(f)

        for label in sorted(mapping.keys()):

            Id, name = mapping[label]

            writer.writerow([label, Id, name])



def get_or_create_label(Id, name, log_func):

    mapping = load_label_map(log_func)

    for label, (stored_Id, stored_name) in mapping.items():

        if stored_Id == Id:

            mapping[label] = (Id, name)

            save_label_map(mapping)

            return label

    new_label = max(mapping.keys(), default=0) + 1

    mapping[new_label] = (Id, name)

    save_label_map(mapping)

    return new_label





# ======================================================================

# --------------------- FLET MODERN UI & LOGIC -------------------------

# ======================================================================



def main(page: ft.Page):

    # --- Page Configuration ---

    page.title = "AI Attendance Pro"

    page.theme_mode = ft.ThemeMode.DARK

    page.padding = 0

    page.window.width = 1000

    page.window.height = 700

    page.fonts = {"RobotoMono": "https://github.com/google/fonts/raw/main/ofl/robotomono/RobotoMono-Regular.ttf"}



    # --- UI Callbacks for Backend ---

    def log_to_console(message):

        current_time = time.strftime('%H:%M:%S')

        log_view.controls.append(ft.Text(f"[{current_time}] {message}", font_family="RobotoMono", color=ft.Colors.GREEN_400, size=13))

        page.update()

        log_view.scroll_to(offset=-1, duration=300)



    def show_toast(message, color=ft.Colors.GREEN):

        page.overlay.append(ft.SnackBar(content=ft.Text(message, weight=ft.FontWeight.BOLD), bgcolor=color, behavior=ft.SnackBarBehavior.FLOATING, open=True))

        page.update()



    ensure_attendance_file(log_to_console)



    # --- Core Tasks ---

    def open_excel_task(e):

        try:

            log_to_console("Opening Attendance Excel file...")

            os.startfile(ATTENDANCE_FILE)

        except AttributeError:

            subprocess.call(['open', ATTENDANCE_FILE])

        except Exception as e:

            show_toast(f"Error: Could not open file.", ft.Colors.RED_500)



    def train_model_task():

        if not os.path.exists(CASCADE_PATH):

            show_toast("Cascade file missing!", ft.Colors.RED_500)

            return



        label_map = load_label_map(log_to_console)

        if not label_map:

            show_toast("No students found in map!", ft.Colors.RED_500)

            return



        try:

            recognizer = cv2.face.LBPHFaceRecognizer_create()

        except AttributeError:

            show_toast("Install opencv-contrib-python!", ft.Colors.RED_500)

            return



        faces, labels = [], []

        valid_images = 0



        log_to_console("Training started. Please wait...")

        for filename in os.listdir(TRAINING_IMAGE_DIR):

            if not filename.lower().endswith(".jpg"): continue

            try:

                parts = filename.split("_")

                if len(parts) < 3: continue

                label = int(parts[2])

                img_path = os.path.join(TRAINING_IMAGE_DIR, filename)

                img = Image.open(img_path).convert('L')

                image_np = np.array(img, 'uint8')

               

                faces.append(image_np)

                labels.append(label)

                valid_images += 1

            except Exception as e:

                continue



        if valid_images == 0:

            show_toast("No valid training images found!", ft.Colors.RED_500)

            return



        recognizer.train(faces, np.array(labels))

        recognizer.save(os.path.join(TRAINING_IMAGE_LABEL_DIR, "Trainer.yml"))

        log_to_console(f"Success! Model trained on {valid_images} images.")

        show_toast(f"Model trained on {valid_images} images!", ft.Colors.GREEN_600)



    def take_images_task():

        Id = txt_id.value.strip()

        name = txt_name.value.strip()

       

        if not Id or not name:

            show_toast("Enter both ID and Name!", ft.Colors.RED_500)

            return

        if "_" in Id or "_" in name:

            show_toast("No underscores allowed in Name or ID.", ft.Colors.RED_500)

            return



        detector = cv2.CascadeClassifier(CASCADE_PATH)

        label = get_or_create_label(Id, name, log_to_console)

        cam = cv2.VideoCapture(0)

       

        if not cam.isOpened():

            show_toast("Cannot open camera!", ft.Colors.RED_500)

            return



        sample_num = 0

        log_to_console(f"Starting camera for {name} (ID: {Id})...")

        window_name = "Taking Images - Press Enter to stop"

       

        try:

            while True:

                ret, frame = cam.read()

                if not ret: break

               

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = detector.detectMultiScale(gray, 1.3, 5)

               

                for (x, y, w, h) in faces:

                    sample_num += 1

                    filename = f"{name}_{Id}_{label}_{sample_num}.jpg"

                    cv2.imwrite(os.path.join(TRAINING_IMAGE_DIR, filename), gray[y:y+h, x:x+w])

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    cv2.putText(frame, f"Capturing {sample_num}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

               

                cv2.imshow(window_name, frame)

                if cv2.waitKey(1) == 13 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1 or sample_num >= 50:

                    break

        finally:

            cam.release()

            cv2.destroyAllWindows()

       

        if sample_num > 0:

            log_to_console(f"Successfully saved {sample_num} images for {name}.")

            show_toast(f"{sample_num} images captured.", ft.Colors.GREEN_600)

        else:

            show_toast("No faces were detected.", ft.Colors.ORANGE_500)



    def track_images_task():

        trainer_file = os.path.join(TRAINING_IMAGE_LABEL_DIR, "Trainer.yml")

        if not os.path.exists(trainer_file):

            show_toast("Train the model first!", ft.Colors.RED_500)

            return



        try:

            recognizer = cv2.face.LBPHFaceRecognizer_create()

        except AttributeError:

            show_toast("Install opencv-contrib-python!", ft.Colors.RED_500)

            return

           

        recognizer.read(trainer_file)

        detector = cv2.CascadeClassifier(CASCADE_PATH)

        mapping = load_label_map(log_to_console)



        cam = cv2.VideoCapture(0)

        if not cam.isOpened():

            show_toast("Cannot open camera!", ft.Colors.RED_500)

            return



        try:

            wb = load_workbook(ATTENDANCE_FILE)

            ws = wb.active

        except PermissionError:

            show_toast("Close Attendance.xlsx first!", ft.Colors.RED_500)

            return



        seen_labels = set()

        font = cv2.FONT_HERSHEY_SIMPLEX

        log_to_console("Live Tracking started. Press ENTER to stop.")

        window_name = "Tracking - Press Enter to stop"



        try:

            while True:

                ret, frame = cam.read()

                if not ret: break

               

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = detector.detectMultiScale(gray, 1.3, 5)

               

                for (x, y, w, h) in faces:

                    label_pred, conf = recognizer.predict(gray[y:y+h, x:x+w])

                   

                    if conf < 70:

                        Id, name = mapping.get(label_pred, ("Unknown", "Unknown"))

                        if label_pred not in seen_labels:

                            seen_labels.add(label_pred)

                            date = datetime.date.today().strftime("%Y-%m-%d")

                            timeStamp = datetime.datetime.now().strftime("%H:%M:%S")

                            ws.append([Id, name, date, timeStamp])

                            log_to_console(f"Attendance marked: {name} ({Id})")

                            try:

                                wb.save(ATTENDANCE_FILE)

                                show_toast(f"Marked: {name}", ft.Colors.GREEN_500)

                            except PermissionError:

                                log_to_console("Warning: Excel file is open.")

                       

                        cv2.putText(frame, f"{Id}-{name}", (x, y-10), font, 0.8, (0, 255, 0), 2)

                    else:

                        cv2.putText(frame, "Unknown", (x, y-10), font, 0.8, (0, 0, 255), 2)

                   

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

               

                cv2.imshow(window_name, frame)

                if cv2.waitKey(1) == 13 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:

                    break

        finally:

            cam.release()

            cv2.destroyAllWindows()

            try: wb.save(ATTENDANCE_FILE)

            except: pass

            log_to_console("Tracking session ended.")



    # --- Thread Launchers ---

    def btn_take_images_click(e): threading.Thread(target=take_images_task, daemon=True).start()

    def btn_train_model_click(e): threading.Thread(target=train_model_task, daemon=True).start()

    def btn_track_images_click(e): threading.Thread(target=track_images_task, daemon=True).start()



    # --- UI COMPONENTS ---

    txt_id = ft.TextField(label="Student ID", prefix_icon=ft.Icons.BADGE, width=300, border_radius=10)

    txt_name = ft.TextField(label="Student Name", prefix_icon=ft.Icons.PERSON, width=300, border_radius=10)



    log_view = ft.ListView(expand=1, spacing=5, padding=10, auto_scroll=True)

    log_container = ft.Container(

        content=log_view,

        margin=10, padding=10, height=150,

        bgcolor=ft.Colors.WHITE10,

        border_radius=10, border=ft.border.all(1, ft.Colors.WHITE10)

    )



    # --- Views (Pages) ---

    register_view = ft.Column([

        ft.Text("Register New Student", size=24, weight=ft.FontWeight.BOLD),

        ft.Text("Enter details and capture facial data.", color=ft.Colors.WHITE70),

        ft.Divider(height=20, color=ft.Colors.TRANSPARENT),

        txt_id, txt_name,

        ft.Divider(height=10, color=ft.Colors.TRANSPARENT),

        ft.Row([

            ft.ElevatedButton("Take Images", icon=ft.Icons.CAMERA_ALT, on_click=btn_take_images_click,

                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10), padding=20)),

            ft.FilledButton("Train Model", icon=ft.Icons.MEMORY, on_click=btn_train_model_click,

                            style=ft.ButtonStyle(bgcolor=ft.Colors.ORANGE_600, shape=ft.RoundedRectangleBorder(radius=10), padding=20))

        ])

    ], alignment=ft.MainAxisAlignment.START, expand=True)



    track_view = ft.Column([

        ft.Text("Live Attendance", size=24, weight=ft.FontWeight.BOLD),

        ft.Text("Start the camera to begin marking attendance automatically.", color=ft.Colors.WHITE70),

        ft.Divider(height=30, color=ft.Colors.TRANSPARENT),

        ft.FilledButton("START TRACKING", icon=ft.Icons.PLAY_ARROW, on_click=btn_track_images_click,

                        style=ft.ButtonStyle(bgcolor=ft.Colors.RED_600, shape=ft.RoundedRectangleBorder(radius=10), padding=25))

    ], alignment=ft.MainAxisAlignment.START, expand=True)



    content_area = ft.AnimatedSwitcher(

        content=register_view,

        transition=ft.AnimatedSwitcherTransition.FADE,

        duration=500,

        reverse_duration=500,

        switch_in_curve=ft.AnimationCurve.EASE_OUT,

        switch_out_curve=ft.AnimationCurve.EASE_IN

    )



    # --- Navigation Handling ---

    def on_nav_change(e):

        if e.control.selected_index == 0:

            content_area.content = register_view

        elif e.control.selected_index == 1:

            content_area.content = track_view

        page.update()



    rail = ft.NavigationRail(

        selected_index=0,

        label_type=ft.NavigationRailLabelType.ALL,

        min_width=100,

        min_extended_width=200,

        group_alignment=-0.9,

        destinations=[

            ft.NavigationRailDestination(icon=ft.Icons.PERSON_ADD_OUTLINED, selected_icon=ft.Icons.PERSON_ADD, label="Register"),

            ft.NavigationRailDestination(icon=ft.Icons.FACT_CHECK_OUTLINED, selected_icon=ft.Icons.FACT_CHECK, label="Attendance"),

        ],

        on_change=on_nav_change,

    )



    # --- Layout Assembly ---

    main_layout = ft.Row([

        rail,

        ft.VerticalDivider(width=1),

        ft.Column([

            ft.Container(

                content=ft.Row([

                    ft.Icon(ft.Icons.FINGERPRINT, size=30, color=ft.Colors.BLUE_400),

                    ft.Text("AI Attendance System", size=20, weight=ft.FontWeight.W_800),

                    ft.Container(expand=True),

                    ft.ElevatedButton("View Excel", icon=ft.Icons.TABLE_CHART, on_click=open_excel_task, color=ft.Colors.GREEN_400)

                ]),

                padding=20

            ),

            ft.Container(content=content_area, expand=True, padding=ft.padding.only(left=40, top=20)),

            ft.Text("   System Console", weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE54),

            log_container

        ], expand=True)

    ], expand=True)



    page.add(main_layout)

    log_to_console("System Initialized. UI engine loaded.")



if __name__ == "__main__":

    ft.run(main)
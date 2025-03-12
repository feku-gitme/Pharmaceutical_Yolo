##### Candy Calorie Counter #####

# Author: Evan Juras, EJ Technology Consultants, https://ejtech.io

# Description:
# This script uses a custom YOLO candy detection model to locate and identify pieces of candy in a live camera view.
# It references a dictionary storing nutritional info about each piece of candy, and tallies the total amount
# of sugar and calories from the candy present in the camera's view.

# Import necessary packages
import os
import sys
import cv2
from ultralytics import YOLO

# Define path to model and other user variables
model_path = 'yolo11s_candy_model.pt'  # Path to model
min_thresh = 0.50                      # Minimum detection threshold
cam_index = 0                          # Index of USB camera
imgW, imgH = 1024, 600                 # Resolution to run USB camera at
record = False                         # Record result video

# Create dictionary to hold info about candy calories and sugar. Each entry is stored as {'candy_type': [calories, grams sugar]}
# These were taken from the Costco candy packaging: https://www.costco.com/.product.100333887.html and https://www.costco.com/.product.100688986.html
brand_info = {
    'Alaxan FR': ['Ibuprofen + Paracetamol', '200 mg/325 mg Capsule', 'pain relief, fever reduction, anti-inflammatory, and combination benefits.', 'Adults and 12 years old and above: Take 1 capsule every 6 hours as needed or as directed by a doctor.'],
    'Alnix': ['Cetirizine Dihydrochloride', '10 mg Tablet', 'Allergic Rhinitis, allergic conjunctivitis, urticaria (hives), other allergic reactions, and cold symptoms.', 'Taken orally (by mouth) every 12 hours, or, as directed by a doctor.'],
    'Ascof Forte': ['Vitex negundo L. (Lagundi Leaf)', '600 mg Tablet', 'Cough relief, expectorant, anti-inflammatory, natural ingredients, support for respiratory health, and mild antipyretic.', 'Adults 3 times a day four times daily. Children 7-12 years: 300mg 3 times a day to four times daily.'],
    'Bioflu': ['Phenylephrine HCI + Chlorphenamine Maleate + Paracetamol', '10 mg/2 mg/500 mg Tablet', 'Fever reduction, pain relief, nasal congestion relief, cough relief, and allergy symptom relief.', 'Adults and Children 12 years and above: Orally, 1 tablet every 6 hours, or as recommended by a doctor.'],
    'Biogesic': ['Paracetamol', '500 mg Tablet', 'Pain relief, fever reduction, post-surgical pain, cold and flu symptoms.', '1 to 2 tablets every 4 to 6 hours, as needed.'],
    'Broxan 30': ['Ambroxol Hydrochloride', '30 mg Tablet', 'Mucolytic agent, cough relief, bronchial conditions, post-surgical care, and respiratory infections.', 'Adults and children over 12 year: One tablet (30 mg ) taken three times daily. Children aged 6 to 12 years: Half a tablet (15 mg) taken three times daily.'],
    'Buscopan': ['Hyoscine Butyl Bromide', '', '20 mg Tablet', 'Relief from abdominal pain, menstrual Pain Relief, urinary tract spasms, preoperative use, and diagnostic procedures.', 'Adults and Children aged 12 years and over is 2 tablets, takes 4 times a day. For Children aged 6 to 11 years, the usual does is 1 tablet, taken 3 times a day.'],
    'Ceticit': ['Cetirizine Hydrochloride', '10 mg Tablet', 'Allergic rhinitis, allergic conjunctivitis, urticaria (hives), and other allergic reactions.', 'Adults and children 6 years and older: 10mg once daily. Children aged 2 to 5 years: 5 mg once daily or 2.5 mg twice daily.'],
    'Datab': ['Loperamide Hydrochloride', '2 mg Capsule', 'Relief from abdominal pain and cramps, reduction of gas and bloating, treatment of irritable bowel syndrome (IBS), flatulence, and digestive discomfort. ', 'Active Ingredients B', 'For aduls 1 to 2 capsule up to 3 times a day and For Children, dosages my vary based on the age and weight, but follow your doctors instruction.'],
    'Decolgen Forte': ['Phenylpropanolamine Hydrochloride + Chlorphenamine Maleate + Paracetamol', '2mg/ 25mg/ 500 mg Tablet', 'Relief for nasal congestion, management of cold symptoms, treatment of allergic rhinitis, sinusitis relief, headache relief.', 'Adults and children 12 years and older: Orally, 1 tablet every 6 hours, or, as recommended by a doctor.'],
    'Decolsin': ['Dextromethorpan HBR + Phenylpropanolamine Hydrochloride + Paracetamol', '15 mg/ 25 mg/ 500 mg', 'Used to relief of clogged nose, postnasal drip, and fever associated with common cold, sinusitis and flu.', 'Adults and Children >12y/o: 1 capsule every 6 hours.'],
    'Diatabs': ['Loperamide Hydrochloride', '2 mg', 'Control and symptomatic relief of acute non specific diarrhea, chronic diarrhea associated with inflammatory bowel disease.', 'Adul dose: Take 2 capsules initially followed by 1 capsule after each loose bowel movement. or, as directed by a doctor.'],
    'Fluimicil': ['Acetylcysteine', '600 mg', 'Used as a mucolytic, meaning it helps to break down and thin mucus in the respiratory tract. This makes it easier to cough up phlegm and clear congestion.', 'Adults: One 200 mg tablet or sachet three times daily, or as directed by a physician.'],
    'Gardan': ['Mefenamic Acid', '500 mg', 'Gardan is used in mild to moderate pain including headache, dental pain, post-operative and post partum pain,dysmonerrhea, menorrhagia, in musculoskeletal and joint disorders.', 'Adult and Children >12 y/o initially 1 tab followed by 1/2 tab every 8 hr as needed but not > 7days.'],
    'Haemorex': ['Tranemic Acid', '500 mg', 'This medication is used short term in people with a certain bleeding disorder to prevent and reduce bleed having a toot puled.', '1 to 1.5 mg 0r 5-10 ml by slow intavenous injection at a rate of 1 ml/minute, two to three times daily.'],
    'Harvimide': ['Loperamide', '2 mg', 'Used to treat diarrhea, decreases the speed at which gut content move,decreases the freaquency of bathroom visits.', 'For adults, start with 4 mg daily, adjusting as necessary until achieving 1-2 solid stools per day, typically with a maintenance dose ranging from 2 to 12 mg daily'],
    'Hyopan': ['Hyoscine Butyl Bromide', '10 mg', 'It is used to relieve nausea, vomiting and dizziness associated with the motion sickness and recovery surgery, also used to treat parkinsonism, spatic muscles states, irritable bowel syndrome, diverticulities, and other conditions.', 'Adults and Children over 12 years: Take 2 tablets of 10 mg or 1 table of 20 mg, 4 times a day. For irritable bowel syndrome, start with 1 tablet of 10 mg, 3 times a day with possible adjustments if needed.'],
    'Imodium': ['Loperamide HCL', '2 mg', 'Control and symptomatic relief of acute nonspecific diarrhea and of chronic diarrhea associated with inflammatory bowel disease.', 'Adults: The recommended initial dose is 4 mg (two capsulees) followed by 2 mg (one capsule) after each unformed stool.'],
    'Kiddelets': ['Paracetamol', 'Adult and children>12 yrs 3-4 tabs every 4hours. Children 7-12 yr 2-3 tab every 4hr, 4-6 yr 1-2 tab every 4hr, 2-3 1 tab every 4 hours.', 'Temporary relief of minor aches & pains e.g, toothache, menstrual cramps, muscular aches, minor arthritis pain & pain associated w/ common cold & flu.', 'This medicine should be taken orally every 4 hours, as needed for pain and/or fever, or, as directed by a doctor. A child should not exceed 5 doses in each 24 hour period. An adult should not take more than 4 g in each 24 hour'],
    'Kremil-S' :['Aluminum Hydroxide, Magnesium Hydroxide and Simethicone', '1-2 tablets after each meal and at bedtime', 'Kremil S is a drug commonly used to relieve symptoms of peptic ulcer, gastritis, esophagitis, and dyspepsia. This medication also relieves gas symptoms, including postoperative gas pain associated with acid reflux.', 'Tablets may be chewed then swallowed with or without water.'],
    'Lormide': ['Loperamide Hydrocholoric', 'Adult: initial dosage 4 mg followed by 2 mg.', 'Relief of acute diarrhea', 'take with full glass of water, with or without food. Follow dosage instructions; do not exceed the recommended amount.'],
    'Losaar 50': ['Losartan Potassium', 'Adults: 50 mg once daily. Children: Consult a healtrhcare professional for appropriate dosing based on age and weight.', 'Hypertension, Heart Failure', 'Take with or without food,  at the same time each day.'],
    'Mecid': ['Mefenamic Acid', '500 mg', 'Used in mild to moderate pain including headache, dental pain, postoperative and postpartum pain, dysmenorrhea, menorrhagia, in musculoskeletal and joint disorders suvh as osteoarthritis and rheumatoid arthritis; and in children with fever and juvenile idiopathic arthritis.', 'Adult: 500 mg should be given to adults up to 3 times (1.5g total) per day. Infants over 6 months: 25 mg/kg of body weight daily in divided dose for not longer than 7 days.'],
    'Medicol Advance': ['Ibuprofen', '400 mg/ 200 mg', 'used for the treatment of different types of pain like headache, migraine, toothache, dysmenorrhea, body pains, and arthritis.',  'Adults and teenagers: 400 mg every 4-6 hours.'],
    'Megyxan': ['Ibuprofen', '500 mg', 'For the relief of acute and chronic pain icluding muscular rheumatic pain, traumatic, dental, post operative and post partum pain, headache and fever.', 'Adults: The usual dose is 500mg initially, followed by 250 mg every 6 hours as needed. The maximum recommended daily dose is typically 1g per day.'],
    'Midol': ['Ibuprofen', '200 mg', 'For menstrual cramping and other effects related to premenstrual syndrome and menstruation.', 'Adults: 2 caplets every 6hrs; max 6/day. Children < 12yrs: consult physician.'],
    'Moxylor': ['Amoxicillin', '500 mg', 'This medication is used to treat a variety of bacterial infections such as infections of the throat, ear, nasal sinuses, respiratory tract, urinary tract, skin and typhoid fever.', 'd'],
    'Mucotoss Forte': ['Paracematol + Guaifenesin + Phenylpropanolamine HCI + Dextromethorphan Hydrobromide + Chlorphenamine Maleate', '325 mg/ 50 mg/ 12.5 mg/ 10 mg/ 1 mg', 'For control of cough associated with colds, allergy and inhalations of irritating substances and psychogenic cough.', 'Adult and Children over 12 yrs and older: 1 capsule 3times a day or as prescribed by the physician.'],
    'Muskelax': ['Ibuprofen + Paracetamol', '500 mg', 'Used for the relief of mild to moderately severe pain of musculoskeletal origin suh as muscle pain, athritis, and rheumatism.', 'Adults and Children 12yrs and older: 1 tablet every 6hrs as needed.'],
    'Nasathera': ['Phenylepropanolamine Hydrochloride + Paracetamol', '25 mg/ 325mg', 'Adults: 1 capsule every 8hrs; Children bet. 6 to 12yrs: 1 capsule at bedtime.'],
    'Neozep Z+ Forte': ['Phenylephaine HCI, Chlorphenamine Maleate, Paracetamol + Zinc', '10 mg/2 mg/325 mg/ 10 mg Tablet', 'Relief for clogged nose, post nasal drip, itchy and watery eyes, sneezing, headache, body aches, and fever associated with the common cold, allergic rhinitis, sinusitis, flu, and other minor repiratory tract infections.'],
    'Pirox': ['Piroxicam', '20 mg', 'Used for relief of the signs and symptoms of osteoarthritis, rheumatoid arthritis.', 'Adults and Children: 1 tablet per day'],
    'Piroxicam': ['Piroxicam', '20 mg', 'Used to treat inflammation caused by osteoarthritis or rheumatoid arthritis.', 'Once or twice a day.'],
    'Plemex Forte': ['Vitex negundo L.', '600 mg', 'Used to treat cough, stuffy nose and chest congestion caused by allergies.', '1 capsule 3-4 times a day every 8hrs.'],
    'Ranzole': ['Omeprazole', '40 mg', 'Used in treatment of heartburn, acid reflux and peptic ulcer disease.', 'Take it preferably on an empty stomach atleast 1hr before a meal.'],
    'Ponstan': ['Mefinamic Acid', '250 mg', 'Used to relieve moderately severe pain such as muscular aches and pains, menstrual cramps, headaches, and dental pain.', 'Active Ingredients B', 'Instructions B'],
    'Rexidol Forte': ['Paracetamol + Caffeine', '500 mg/ 65 mg', 'To relief moderate to severe pain, including musculoskeletal pain, and migrains.', 'Adults and Children 12yrs and above: 1-2 tablets every 6hrs, as needed for pain.'],
    'Robitussin': ['Guaifenesin', '200 mg', 'Used to reduce chest congestion caused by common cold, infections or allergies.', 'Adults and Children 12yrs and older: 1 capsule every 6-8 hrs as needed.'],
    'Saphlecox 200': ['Cefixime', '200 mg', 'Treatment of bacterial infections, including respiratory tract infections, urinary tract infections and ear infections', 'Adults: 1 tablet every 12hrs; Children: varies based on age and weight.'],
    'Saphmirate-T50': ['Butamirate Citrate', '50 mg', 'Used to suppress dry, non-productive coughs associated with respiratory conditions like pneumonia.', 'Adults: 2-3 tabs daily at interval of 8-12 hours.'],
    'Saphroxol C75': ['Ambroxol Hydrochloride', '75 mg', 'Used for secretory therapy in acute and chronic bronchopulmonary disease associated with abnormal mucus secretion and impared mucus transport.', '1 capsule a day'],
    'Saridon': ['Paracetamol + Propyphenazone + Caffeine', '500 mg/ 150 mg/ 50 mg', 'For fast and effective relief of mild to severe headache, tootheache, menstrual discomfort, postoperative and rheumatic pain.', 'Adults: 1-2 tab/day; Adolescent aged 12-16yrs: 1 tab/day.'],
    'Skelan 550': ['Naphroxen Sodium', '550 mg', 'Used for management of pain including headache, migraine, post operative pain, post partum pain and primary dysmonorrhea.', 'Adult: 1 tab every 12hrs; Children 12yr: 1 tablet for every 8-12 hrs'],
    'Solmux': ['Carbocisteine', '500 mg', 'Used to treat cough with phlegm.', 'Taken every 8hrs or as recommended by doctors.'],
    'Solmux Advance': ['Carbocisteine + Zinc', '500mg + 5mg', 'Relief of cough with respiratory tract disorders such as acute bronchitis.', 'Adults and Childrens 12yrs and older: 1 tablet every 8 hours.'],
    'Solmux Broncho': ['Salbutamol Sulfate + Carbocisteine', '2 mg/ 500 mg', 'For treatment of productive cough associated with airway disorders, such as acute and chronic bronchitis, bronchial asthma and bronchiectasis.', 'Can swallow the capsule with or without chewing or dissolving it in liquid.'],
    'Symdex-D': ['Paracetamol + Phenylpropanolamine hydrochloride + Chlorphenamine maleate', '500 mg/ 25 mg/ 2mg', 'For common colds, allergic rhinitis; Sinusitis; Nasal decongestant', 'Adult: 1 tab every 6hrs; Children (7-12 yrs old) one-half tablet every 6hrs.'],
    'Tuseran Forte': ['Dextromethorphan Hydrobromide + Phenylephrine Hydrochloride + Paracetamol', '15 mg/ 10 mg/ 325 mg', 'Used to relief cough, clogged nose, ponstnasal drip, headache, body aches and fever.', 'Adults and Children 12yrs and older: 1 capsule every 6 hrs.'],
    'Ventrex-G': ['Salbutamol Guaitenesin', '2 mg', 'Used to relieve and prevent breathing difficulties in conditions like asthma.', '2-3mg 3-4 times daily may be increased up to max of 8mg 3-4 times daily.'],
    'Zosec': ['Omperazole', '20 mg', 'Used in treatment of acidity, heartburn, acid reflux and peptic ulcer.', 'once a day (every 24hrs) for 14 days before eating'],
    'BrandB': ['Generic Name B', 'DosageB', 'UsesB', 'Active Ingredients B', 'Instructions B'],
    
    
    # Add more brands as needed
}

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('WARNING: Model path is invalid or model was not found.')
    sys.exit()

# Load the model into memory and get label map
model = YOLO(model_path, task='detect')
labels = model.names

# Initialize camera
cap = cv2.VideoCapture(cam_index)
ret = cap.set(3, imgW)  # Set width
ret = cap.set(4, imgH)  # Set height

# Set up recording
if record:
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (imgW, imgH))

# Set bounding box colors (using the Tableau 10 color scheme)
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), 
               (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241), 
               (98, 118, 150), (172, 176, 184)]

# Begin inference loop
while True:
    # Grab frame from camera
    ret, frame = cap.read()
    if frame is None or not ret:
        print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
        break

    # Run inference on frame with tracking enabled
    results = model.track(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable to hold detected brands
    brands_detected = []

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()  # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze()  # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int)  # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > min_thresh:
            # Draw box around object
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw label for object
            label = f'{classname}: {int(conf * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Add object to list of detected brands
            brands_detected.append(classname)

    # Process list of brands that have been detected to display their information
    for brand_name in brands_detected:
        if brand_name in brand_info:
            generic_name, dosage, uses, instructions = brand_info[brand_name] #active_ingredients,
            # Display brand information on the frame
            cv2.putText(frame, f'Brand: {brand_name}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Generic Name: {generic_name}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Dosage: {dosage}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Uses: {uses}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #cv2.putText(frame, f'Active Ingredients: {active_ingredients}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Instructions: {instructions}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display results
    cv2.imshow('Brand detection results', frame)  # Display image
    if record:
        recorder.write(frame)  # Record frame to video (if enabled)

    # Poll for user keypress and wait 5ms before continuing to next frame
    key = cv2.waitKey(5)

    if key == ord('q') or key == ord('Q'):  # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'):  # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):  # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png', frame)

# Clean up
cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()

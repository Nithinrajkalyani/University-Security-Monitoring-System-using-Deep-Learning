import psycopg2
import os
from PIL import Image
import io

# Connect to the database
conn = psycopg2.connect(host="localhost", port=5432, dbname="face_detection", user="postgres", password="raj@123")

# Define the query to retrieve all image data
query = "SELECT srn,image_name,image_data FROM known_persons"

# Execute the query
cur = conn.cursor()
cur.execute(query)
kk2=[]

# Fetch all results
results = cur.fetchall()

# Loop through each result
for result in results:
    srn = str(result[0])
    image_name = str(result[1])
    image1 = list(map(list,image_name))
    t11 = image1[3:-4]
    kk1=[]
    for pp in t11:
        kk1 = kk1 + pp
        kk2 = ''.join(kk1)
    image_data = result[2]

    # Decode the binary data
    image_binary = io.BytesIO(image_data)
    image = Image.open(image_binary)
    filename = kk2 + srn + ".jpg"

    # Save the image file to disk
    image.save(os.path.join("C:/Mini/Attendance/decode_known_person", filename))

# Close the database connection and cursor
cur.close()
conn.close()

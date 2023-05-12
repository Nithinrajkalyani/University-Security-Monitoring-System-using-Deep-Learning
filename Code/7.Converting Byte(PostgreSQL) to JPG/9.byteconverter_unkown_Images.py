import psycopg2
import os
from PIL import Image
import io

# Connect to the database
conn = psycopg2.connect(host="localhost", port=5432, dbname="face_detection", user="postgres", password="raj@123")

# Define the query to retrieve all image data
query = "SELECT srn, image_data FROM unknown_persons"

# Execute the query
cur = conn.cursor()
cur.execute(query)

# Fetch all results
results = cur.fetchall()

# Loop through each result
for result in results:
    srn = str(result[0])
    image_data = result[1]
    
    # Decode the binary data
    image_binary = io.BytesIO(image_data)
    image = Image.open(image_binary)
    filename = "Unknown_" + srn + ".jpg"

    # Save the image file to disk
    image.save(os.path.join("C:/Mini/Attendance/decode_unknown_person", filename))

# Close the database connection and cursor
cur.close()
conn.close()

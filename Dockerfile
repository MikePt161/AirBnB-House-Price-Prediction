# The base image
FROM python:3.9.5

# Main working dir for subsequent commands
WORKDIR /app

# Copy the file containing the necessary python libraries and install them to the image
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the saved model and the web apps
# COPY src /app/src
COPY server.py /app/server.py
COPY preprocessing.py /app/preprocessing.py

COPY data/amenities_cat.csv data/amenities_cat.csv
COPY data/fill_values.csv data/fill_values.csv
COPY data/neighbourhoods.csv data/neighbourhoods.csv
COPY data/rf.joblib data/rf.joblib
COPY data/room_types.csv data/room_types.csv
COPY data/scaler.joblib data/scaler.joblib



# Run the server 
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

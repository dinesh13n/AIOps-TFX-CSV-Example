import os
import csv
import tensorflow as tf

# Get the RAW data file path
data_file = os.path.join("Data","all_tickets.csv")

#create directory for the tf_record data
os.makedirs("Data\\tf_record_data",exist_ok=True)

#Create object for the tf record file
tfx_record_file = tf.io.TFRecordWriter(os.path.join("Data","tf_record_data","tf_record_file"))

def _bytes_feature(value):
    value = value.encode()
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _Int64_feature(value):
    value = int(value)
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _float_feature(value):
    value = float(value)
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

with open(data_file,encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",", quotechar='"')
    col_list = ['\ufeffSummary','Owner_Group']
    for row in csv_reader:
        feature = {
            col_list[0]:_bytes_feature(row[col_list[0]]),
            col_list[1]:_bytes_feature(row[col_list[1]])
        }

        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        tfx_record_file.write(example.SerializeToString())

tfx_record_file.close()

if __name__ == '__main__':
    print("Processing started")

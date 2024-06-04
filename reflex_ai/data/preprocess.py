import csv

schema = {'transcript_id':0,'mi_quality':1,'video_title':2,'video_url':3,'topic':4,'utterance_id':5,'interlocutor':6,'timestamp':7,
          'utterance_text':8,'main_therapist_behaviour':9,
          'client_talk_type':10}

with open('AnnoMI_simple_test.csv', 'r') as file:
    reader = csv.reader(file)
    full_rows = []
    previous_row=" "
    for ind, row in enumerate(reader):
        if ind == 0:
            header = row
            header.append("client utterance")
            continue
        if row[schema['interlocutor']] == "therapist":
            full_rows.append(row)
            if row[schema['utterance_id']] != 0:
                # if previous_row != []:
                full_rows[-1].extend([previous_row])
                # print("full rows", full_rows)
            previous_row = " "
        else:
            previous_row = row[schema['utterance_text']]
            # print("prev",previous_row)
        # if ind == 150:
        #     break

    print(full_rows)
    print(len(full_rows))
    # print(row[schema['utterance_id']])
    # break

with open("AnnoMI-processed_test.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(header)
    csvwriter.writerows(full_rows)

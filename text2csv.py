from itertools import islice

d_path = r'data/kaggle_visible_evaluation_triplets.txt'
csv_path = r'data/song_data.csv'

with open(d_path, 'r') as f1, open(csv_path, 'w') as f2:
    i = 0
    f2.write('user_id,song_id,listen_count\n')
    while True:
        next_n_lines = list(islice(f1, 9))
        if not next_n_lines:
            break

        # process next_n_lines: get user_id, song_id, listen_count info
        output_line = ''
        for line in next_n_lines:
            user_id, song_id, listen_count = line.split('\t')
            output_line += '{},{},{}\n'.format(user_id, song_id, listen_count.strip())
        f2.write(output_line)

        # print status
        i += 1
        if i % 20000 == 0:
            print('{} songs formated'.format(i))

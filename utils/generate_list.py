
import os



def generate_small_list():
    data_path = 'D:/dataset/kientics/small_kinetics/rgb'
    new_list = open('small_kinetics_rgb.list', 'w')

    all_class = os.listdir(data_path)
    all_class.sort()

    for i in range(len(all_class)):
        print(i)
        cur_class = all_class[i]
        cur_class_path = os.path.join(data_path, cur_class)

        all_sample = os.listdir(cur_class_path)

        for j in range(len(all_sample)):
            cur_sample_id = all_sample[j]
            cur_sample_path = os.path.join(cur_class_path, cur_sample_id)

            all_frames = os.listdir(cur_sample_path)

            frame_num = str(len(all_frames))

            new_list.write(cur_class+'/'+cur_sample_id + ' ' + frame_num + ' ' + str(i) + '\n')

def generate_complete_list():

    ori_lines = open('E:/code/pycharm_projects/prepare_kinetics_data/kinetics_rgb_train_v2.list')
    ori_lines = list(ori_lines)

    new_list = open('kinetics_400.list','w')

    for i in range(len(ori_lines)):
        line = ori_lines[i].strip('\n').split()
        sample_dir = line[0].split('/')
        num_frames = line[1]
        label = line[2]

        video_id = sample_dir[-1]
        cur_class = sample_dir[-2]


        new_list.write(cur_class+'/'+video_id+' ' + num_frames + ' ' + label + '\n')




if __name__ == '__main__':
    #generate_small_list()
    generate_complete_list()


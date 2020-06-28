import cv2
from utils.show_flow import computeImg
import numpy as np


def show_pattern_1(img):



    cv2.line(img, (0, 28), (112, 28), color=(0, 255, 0))
    cv2.line(img, (0, 56), (112, 56), color=(0, 255, 0))
    cv2.line(img, (0, 84), (112, 84), color=(0, 255, 0))

    cv2.line(img, (28, 0), (28, 112), color=(0, 255, 0))
    cv2.line(img, (56, 0), (56, 112), color=(0, 255, 0))
    cv2.line(img, (84, 0), (84, 112), color=(0, 255, 0))


    cv2.imshow('pattern 1', img)

    cv2.waitKey()



def show_pattern_2(img):



    cv2.rectangle(img, (14, 14), (98, 98), (0, 255, 0))
    cv2.rectangle(img, (28, 28), (84, 84), (0, 255, 0))
    cv2.rectangle(img, (42, 42), (70, 70), (0, 255, 0))



    cv2.imshow('pattern 2', img)

    cv2.waitKey()


def show_pattern_3(img):

    cv2.line(img, (0, 0), (112, 112), color=(0, 255, 0))
    cv2.line(img, (112, 0), (0, 112), color=(0, 255, 0))
    cv2.line(img, (56, 0), (56, 112), color=(0, 255, 0))
    cv2.line(img, (0, 56), (112, 56), color=(0, 255, 0))

    # cv2.line(img, (0, 56), (112, 56), color=(0, 255, 0))
    # cv2.line(img, (0, 84), (112, 84), color=(0, 255, 0))
    #
    # cv2.line(img, (28, 0), (28, 112), color=(0, 255, 0))
    # cv2.line(img, (56, 0), (56, 112), color=(0, 255, 0))
    # cv2.line(img, (84, 0), (84, 112), color=(0, 255, 0))

    cv2.imshow('pattern 3', img)

    cv2.waitKey()





def show(sample_batched):

    # video clip: 3 x 16 x 112 x 112
    # flow: 15 x 112 x 112


    video_clip, u_flow_15, v_flow_15, motion_labels, du, dv= sample_batched['clip'][0], sample_batched['u_flow'][0], sample_batched['v_flow'][0], \
                                                             sample_batched['motion_label'][0] , sample_batched['du'], sample_batched['dv']



    video_clip = np.transpose(video_clip, (1, 2, 3, 0))


    ## dv
    du_x_sum = du['du_x_sum'][0].numpy()
    du_y_sum = du['du_y_sum'][0].numpy()

    motion_labels = motion_labels.numpy()

    mb_u_sum = computeImg(du_x_sum, du_y_sum)

    print("u pattern 1:", motion_labels[0])
    show_pattern_1(mb_u_sum.copy())

    print("u pattern 2:", motion_labels[4])
    show_pattern_2(mb_u_sum.copy())

    print("u pattern 3:", motion_labels[8])
    show_pattern_3(mb_u_sum.copy())


    ## dv

    dv_x_sum = dv['dv_x_sum'][0].numpy()
    dv_y_sum = dv['dv_y_sum'][0].numpy()

    mb_v_sum = computeImg(dv_x_sum, dv_y_sum)

    print("v pattern 1:", motion_labels[2])
    show_pattern_1(mb_v_sum.copy())

    print("v pattern 2:", motion_labels[6])
    show_pattern_2(mb_v_sum.copy())

    print("v pattern 3:", motion_labels[10])
    show_pattern_3(mb_v_sum.copy())

    ## global

    print("global u:", motion_labels[-2])

    print("global v:", motion_labels[-1])

    print("motion label:", motion_labels[:])


    for i in range(15):

        cur_video_clip = video_clip[i].numpy()
        cv2.imshow("flip img", cur_video_clip)

        cur_u_flow = u_flow_15[i].numpy()
        cur_v_flow = v_flow_15[i].numpy()

        cur_flow = computeImg(cur_u_flow, cur_v_flow)
        cv2.imshow('flow', cur_flow)


        cv2.waitKey()






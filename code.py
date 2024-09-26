import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def get_sift_correspondences(img1, img2):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    # sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()  # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    # Display the matches
    # img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img_draw_match[:, :, ::-1])
    # plt.show()

    return points1, points2

def k_point_sample(points1, points2,k):
    # TODO: Sample K points for Homography Estimation
    return points1, points2

def homography_estimation(points1, points2):
    # TODO: Estimate Homography Matrix

    # This is an example. Don't use this function in your implementation
    return cv.findHomography(points1, points2, cv.USAC_ACCURATE, 1, maxIters=1_00000, confidence=0.99999)[0]

def homography_warping(img1, img2, H):
    # TODO: Image Warping

    # This is an example. don't use this function in your implementation
    return cv.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))



if __name__ == '__main__':
    # usage: python new_1.py img1_path img2_path gt_correspondences_path
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])

    points1, points2 = get_sift_correspondences(img1, img2)


    # Sample K points for Homography Estimation
    points1, points2 = k_point_sample(points1, points2, 4)


    # Estimate Homography Matrix
    H = homography_estimation(points1, points2)

    # TODO: Calculate ERROR Using gt_correspondences


    # Image Warping
    img1_warping = homography_warping(img1, img2, H)
    concate_img = np.zeros((max(img1_warping.shape[0], img2.shape[0]), img1_warping.shape[1] + img2.shape[1], 3), np.uint8)
    concate_img[:img1_warping.shape[0], :img1_warping.shape[1]] = img1_warping
    concate_img[:img2.shape[0], img1_warping.shape[1]:] = img2

    plt.imshow(concate_img[:, :, ::-1])
    plt.show()

    # cv.imshow('Final Image', concate_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


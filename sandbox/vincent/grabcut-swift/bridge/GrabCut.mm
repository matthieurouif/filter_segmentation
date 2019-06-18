//
//  Custom.m
//  bridge
//
//  Created by Vincent on 03/06/2019.
//  Copyright © 2019 Vincent. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "GrabCut.h"

//#import <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>


void generate_trimap(cv::Mat & pred, cv::Mat & trimap) {
    
    // Create trimap container
    trimap = cv::Mat(pred.size(), pred.type());
    
    // Parameters
    const int t0 = 0.05 * 255;
    const int t1 = 0.95 * 255;
    
    // Go through image pixels
    for (int y=0; y<pred.rows; ++y) {
        for (int x=0; x<pred.cols; ++x) {
            
            // Get prediction value
            const int val = pred.at<uchar>(cv::Point(x,y));
            
            // Create the trimap
            if (val <= t0)
                trimap.at<uchar>(cv::Point(x,y)) = cv::GC_BGD;
            else if (val < 127)
                trimap.at<uchar>(cv::Point(x,y)) = cv::GC_PR_BGD;
            else if (val < t1)
                trimap.at<uchar>(cv::Point(x,y)) = cv::GC_PR_FGD;
            else
                trimap.at<uchar>(cv::Point(x,y)) = cv::GC_FGD;
        }
    }
}


void deduce_mask(cv::Mat & trimap, cv::Mat & mask) {
    
    // Create mask container
    mask = cv::Mat(trimap.size(), trimap.type());
    
    // Go through trimap pixels
    for (int y=0; y<trimap.rows; ++y) {
        for (int x=0; x<trimap.cols; ++x) {
            
            // Get prediction value
            const int val = trimap.at<uchar>(cv::Point(x,y));
            
            // Create the trimap
            if (val == cv::GC_BGD || val == cv::GC_PR_BGD)
                mask.at<uchar>(cv::Point(x,y)) = 0;
            else
                mask.at<uchar>(cv::Point(x,y)) = 255;
        }
    }
}


void create_sticker(cv::Mat & image, cv::Mat & mask, cv::Mat & sticker) {
    
    // Create sticker container
    sticker = cv::Mat(image.size(), image.type());
    
    // Fill the sticker
    for (int y=0; y<image.rows; ++y) {
        for (int x=0; x<image.cols; ++x) {
            
            if (mask.at<uchar>(y, x) == 255) {
                sticker.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(y,x);
            }
            else {
                sticker.at<cv::Vec3b>(y, x)[0] = 255;
                sticker.at<cv::Vec3b>(y, x)[1] = 255;
                sticker.at<cv::Vec3b>(y, x)[2] = 255;
            }
        }
    }
    
}




@implementation GrabCut

+ (void) testWithImage: (UInt8*)image
                 width: (int32_t)width
                height: (int32_t)height
           bytesPerRow: (int32_t) bytesPerRow{
    NSLog(@"Coucou mon chien d'amour <3");
    NSLog(@"ObjectiveC : %d, %d, %d", image[30], image[31], image[32]);
    NSLog(@"Image size : %d x %d", width, height);
    
    // Display values
    for (int j=0; j<height; j++) {
        for (int i=0; i<width; i++) {
            UInt8 r = image[j * bytesPerRow + i * 3 + 0];
            UInt8 g = image[j * bytesPerRow + i * 3 + 1];
            UInt8 b = image[j * bytesPerRow + i * 3 + 2];
            NSLog(@"Objective-C : (%d, %d) = (%d,%d,%d)", j, i, r, g, b);
        }
    }
    
    //    // Change values
    //    for (int j=0; j<height; j++) {
    //        for (int i=0; i<width; i++) {
    //            image[j * bytesPerRow + i * 3 + 0] = 0;
    //            image[j * bytesPerRow + i * 3 + 1] = i;
    //            image[j * bytesPerRow + i * 3 + 2] = 33;
    //        }
    //    }
    
    // Create a Mat and fill it
    cv::Mat cvImage = cv::Mat(height, width, CV_8UC3);
    for (int j=0; j<height; j++) {
        for (int i=0; i<width; i++) {
            UInt8 r = image[j * bytesPerRow + i * 3 + 0];
            UInt8 g = image[j * bytesPerRow + i * 3 + 1];
            UInt8 b = image[j * bytesPerRow + i * 3 + 2];
            cv::Vec3b & color = cvImage.at<cv::Vec3b>(cv::Point(i,j));
            color[0] = b;
            color[1] = g;
            color[2] = r;
        }
    }
    
    cv::imshow("input image", cvImage);
    cv::waitKey();
    cv::destroyAllWindows();
    
    // Modify the image using OpenCV
    for (int j=0; j<height; j++) {
        for (int i=0; i<width; i++) {
            cv::Vec3b & color = cvImage.at<cv::Vec3b>(cv::Point(i,j));
            color[0] = MAX(0, MIN(255, j));
        }
    }
    
    
    cv::imshow("image modified by opencv", cvImage);
    cv::waitKey();
    cv::destroyAllWindows();
    
}

+ (void) displayWithImage: (UInt8*)image
                    width: (int32_t)width
                   height: (int32_t)height
              bytesPerRow: (int32_t) bytesPerRow{
    
    // Create a Mat and fill it
    cv::Mat cvImage = cv::Mat(height, width, CV_8UC3);
    for (int j=0; j<height; j++) {
        for (int i=0; i<width; i++) {
            UInt8 r = image[j * bytesPerRow + i * 3 + 0];
            UInt8 g = image[j * bytesPerRow + i * 3 + 1];
            UInt8 b = image[j * bytesPerRow + i * 3 + 2];
            cv::Vec3b & color = cvImage.at<cv::Vec3b>(cv::Point(i,j));
            color[0] = b;
            color[1] = g;
            color[2] = r;
        }
    }
    
    cv::imshow("input image", cvImage);
    cv::waitKey();
    cv::destroyAllWindows();    
}

+ (void) processWithImage: (UInt8*)image
                 imgWidth: (int32_t)imgWidth
                imgHeight: (int32_t)imgHeight
           imgBytesPerRow: (int32_t)imgBytesPerRow
     imgBytesPerComponent: (int32_t)imgBytesPerComponent
               prediction: (UInt8*)prediction
                 preWidth: (int32_t)preWidth
                preHeight: (int32_t)preHeight
           preBytesPerRow: (int32_t)preBytesPerRow
     preBytesPerComponent: (int32_t)preBytesPerComponent {
    
    // Create a Mat and fill it
    cv::Mat cvImage = cv::Mat(imgHeight, imgWidth, CV_8UC3);
    for (int j=0; j<imgHeight; j++) {
        for (int i=0; i<imgWidth; i++) {
            UInt8 r = image[j * imgBytesPerRow + i * imgBytesPerComponent + 0];
            UInt8 g = image[j * imgBytesPerRow + i * imgBytesPerComponent + 1];
            UInt8 b = image[j * imgBytesPerRow + i * imgBytesPerComponent + 2];
            cv::Vec3b & color = cvImage.at<cv::Vec3b>(cv::Point(i,j));
            color[0] = b;
            color[1] = g;
            color[2] = r;
        }
    }
    
    // Create a Mat and fill it
    cv::Mat cvPrediction = cv::Mat(preHeight, preWidth, CV_8UC1);
    for (int j=0; j<preHeight; j++) {
        for (int i=0; i<preWidth; i++) {
            UInt8 p = prediction[j * preBytesPerRow + i];
            uint8 & color = cvPrediction.at<uint8>(cv::Point(i,j));
            color = p;
        }
    }
    
    // Compute the trimap
    cv::Mat trimap;
    generate_trimap(cvPrediction, trimap);
    
    // Copute naive mask
    cv::Mat mask0;
    deduce_mask(trimap, mask0);
    
    // Apply grabcut
    cv::Mat bgd_model;
    cv::Mat fgd_model;
    cv::grabCut(cvImage, trimap, cv::Rect(), bgd_model, fgd_model, 5, cv::GC_INIT_WITH_MASK);
    
    // Compute grabcut mask
    cv::Mat mask1;
    deduce_mask(trimap, mask1);
    
    // Generate stickers
    cv::Mat sticker0, sticker1;
    create_sticker(cvImage, mask0, sticker0);
    create_sticker(cvImage, mask1, sticker1);
    
    // Display
    //    cv::imshow("image", image);
    cv::imshow("mask0", mask0);
    cv::imshow("mask1", mask1);
    cv::imshow("sticker0", sticker0);
    cv::imshow("sticker1", sticker1);
    cv::waitKey();
    cv::destroyAllWindows();
}

@end

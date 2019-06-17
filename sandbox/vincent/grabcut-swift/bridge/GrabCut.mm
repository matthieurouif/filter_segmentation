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
//#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>

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
    
    cv::imshow("Image", cvImage);
    cv::imshow("Predition", cvPrediction);
    cv::waitKey();
    cv::destroyAllWindows();
}

@end

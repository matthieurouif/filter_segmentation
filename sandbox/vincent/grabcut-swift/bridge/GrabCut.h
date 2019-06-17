//
//  Custom.h
//  bridge
//
//  Created by Vincent on 03/06/2019.
//  Copyright © 2019 Vincent. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface GrabCut : NSObject

+ (void) testWithImage: (UInt8*)image
                 width: (int32_t)width
                height: (int32_t)height
           bytesPerRow: (int32_t)bytesPerRow;

+ (void) displayWithImage: (UInt8*)image
                    width: (int32_t)width
                   height: (int32_t)height
              bytesPerRow: (int32_t)bytesPerRow;

+ (void) processWithImage: (UInt8*)image
                 imgWidth: (int32_t)imgWidth
                imgHeight: (int32_t)imgHeight
           imgBytesPerRow: (int32_t)imgBytesPerRow
     imgBytesPerComponent: (int32_t)imgBytesPerComponent
               prediction: (UInt8*)prediction
                 preWidth: (int32_t)preWidth
                preHeight: (int32_t)preHeight
           preBytesPerRow: (int32_t)preBytesPerRow
     preBytesPerComponent: (int32_t)preBytesPerComponent;


@end

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

@end

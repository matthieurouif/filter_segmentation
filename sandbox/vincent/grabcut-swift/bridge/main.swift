//
//  main.swift
//  bridge
//
//  Created by Vincent on 31/05/2019.
//  Copyright © 2019 Vincent. All rights reserved.
//

import Foundation
import AVFoundation
import AppKit


// ---------------------------------
// VERSION 1
// ---------------------------------


//// Image parameters
//let width = 97
//let height = 100
//
//// Create pixel buffer
//var pixelBuffer: CVPixelBuffer?
//let status = CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_24RGB, nil, &pixelBuffer)
//if status != kCVReturnSuccess {
//    print("Error: could not create resized pixel buffer", status)
//}
//
//// Get buffer
//CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags.readOnly)
//let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer!)
//let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)
//
//// Get bytes per row
//let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer!)
//let bpp = bytesPerRow / width
//
//// Display
//print("Width  = \(CVPixelBufferGetWidth(pixelBuffer!))")
//print("Height = \(CVPixelBufferGetHeight(pixelBuffer!))")
//print("bytesPerRow = \(bytesPerRow)")
//print("bpp = \(bpp)")
////exit(0)
//
//// Create red horizontal gradient
//for j in 0..<height {
//    for i in 0..<width {
//        buffer[j * bytesPerRow + i * 3 + 0] = UInt8(i)
//        buffer[j * bytesPerRow + i * 3 + 1] = 0
//        buffer[j * bytesPerRow + i * 3 + 2] = 0
//    }
//}
//
//// Display values
//for j in 0..<height {
//    for i in 0..<width {
//        let r = buffer[j * bytesPerRow + i * 3 + 0]
//        let g = buffer[j * bytesPerRow + i * 3 + 1]
//        let b = buffer[j * bytesPerRow + i * 3 + 2]
//        print("Swift : (\(j), \(i)) = (\(r),\(g),\(b))")
//    }
//}
//
//GrabCut.test(withImage: buffer,
//             width: Int32(width),
//             height: Int32(height),
//             bytesPerRow: Int32(bytesPerRow))
//
//// Display values
//for j in 0..<height {
//    for i in 0..<width {
//        let r = buffer[j * bytesPerRow + i * 3 + 0]
//        let g = buffer[j * bytesPerRow + i * 3 + 1]
//        let b = buffer[j * bytesPerRow + i * 3 + 2]
//        print("Swift2 : (\(j), \(i)) = (\(r),\(g),\(b))")
//    }
//}



// ---------------------------------
// VERSION 2
// ---------------------------------

//let path = "/Users/vincent/Desktop/dog.jpg"
let img_path = "/Users/vincent/data/Work/Projects/filter_segmentation/sandbox/vincent/grabcut-c++/ref_image.png"
let pre_path = "/Users/vincent/data/Work/Projects/filter_segmentation/sandbox/vincent/grabcut-c++/ref_pred.png"

// Read the image
let img = NSImage(contentsOfFile: img_path)
let img_data = img?.tiffRepresentation
let img_bitmap = NSBitmapImageRep(data: img_data!)

// Read the prediction
let pre = NSImage(contentsOfFile: pre_path)
let pre_data = pre?.tiffRepresentation
let pre_bitmap = NSBitmapImageRep(data: pre_data!)

// Image parameters
let img_width = Int(img!.size.width)
let img_height = Int(img!.size.height)

// Prediction parameters
let pre_width = Int(pre!.size.width)
let pre_height = Int(pre!.size.height)

// Create pixel buffer for image
var img_pixelBuffer: CVPixelBuffer?
let status = CVPixelBufferCreate(nil, img_width, img_height, kCVPixelFormatType_24RGB, nil, &img_pixelBuffer)
if status != kCVReturnSuccess {
    print("Error: could not create pixel buffer", status)
}

// Create pixel buffer for prediction
var pre_pixelBuffer: CVPixelBuffer?
let status2 = CVPixelBufferCreate(nil, pre_width, pre_height, kCVPixelFormatType_OneComponent8, nil, &pre_pixelBuffer)
if status2 != kCVReturnSuccess {
    print("Error: could not create pixel buffer", status2)
}

// Get image buffer
CVPixelBufferLockBaseAddress(img_pixelBuffer!, CVPixelBufferLockFlags.readOnly)
let img_baseAddress = CVPixelBufferGetBaseAddress(img_pixelBuffer!)
let img_buffer = img_baseAddress!.assumingMemoryBound(to: UInt8.self)

// Get prediction buffer
CVPixelBufferLockBaseAddress(pre_pixelBuffer!, CVPixelBufferLockFlags.readOnly)
let pre_baseAddress = CVPixelBufferGetBaseAddress(pre_pixelBuffer!)
let pre_buffer = pre_baseAddress!.assumingMemoryBound(to: UInt8.self)

// Get bytes per row for image
let img_bytesPerRow = CVPixelBufferGetBytesPerRow(img_pixelBuffer!)
let img_bytesPerComponent = img_bytesPerRow / img_width

// Get bytes per row for prediction
let pre_bytesPerRow = CVPixelBufferGetBytesPerRow(pre_pixelBuffer!)
let pre_bytesPerComponent = pre_bytesPerRow / pre_width

// Display
print("Image")
print("  Width               = \(CVPixelBufferGetWidth(img_pixelBuffer!))")
print("  Height              = \(CVPixelBufferGetHeight(img_pixelBuffer!))")
print("  bytes per row       = \(img_bytesPerRow)")
print("  Bytes per component = \(img_bytesPerComponent)\n")
print("Prediction")
print("  Width               = \(CVPixelBufferGetWidth(pre_pixelBuffer!))")
print("  Height              = \(CVPixelBufferGetHeight(pre_pixelBuffer!))")
print("  bytes per row       = \(pre_bytesPerRow)")
print("  Bytes per component = \(pre_bytesPerComponent)")

// Copy image data
for j in 0..<img_height {
    for i in 0..<img_width {
        let r = UInt8(255 * Float(img_bitmap!.colorAt(x: i, y: j)!.redComponent))
        let g = UInt8(255 * Float(img_bitmap!.colorAt(x: i, y: j)!.greenComponent))
        let b = UInt8(255 * Float(img_bitmap!.colorAt(x: i, y: j)!.blueComponent))
        img_buffer[j * img_bytesPerRow + i * img_bytesPerComponent + 0] = r
        img_buffer[j * img_bytesPerRow + i * img_bytesPerComponent + 1] = g
        img_buffer[j * img_bytesPerRow + i * img_bytesPerComponent + 2] = b
    }
}

// Copy prediction data
for j in 0..<pre_height {
    for i in 0..<pre_width {
        let p = UInt8(255 * Float(pre_bitmap!.colorAt(x: i, y: j)!.whiteComponent))
        pre_buffer[j * pre_bytesPerRow + i] = p
//        print(p)
    }
}

//GrabCut.display(withImage: img_buffer,
//                width: Int32(img_width),
//                height: Int32(img_height),
//                bytesPerRow: Int32(img_bytesPerRow))

GrabCut.process(withImage: img_buffer,
                imgWidth: Int32(img_width),
                imgHeight: Int32(img_height),
                imgBytesPerRow: Int32(img_bytesPerRow),
                imgBytesPerComponent: Int32(img_bytesPerComponent),
                prediction: pre_buffer,
                preWidth: Int32(pre_width),
                preHeight: Int32(pre_height),
                preBytesPerRow: Int32(pre_bytesPerRow),
                preBytesPerComponent: Int32(pre_bytesPerComponent))


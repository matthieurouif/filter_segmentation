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
let path = "/Users/vincent/data/Work/Projects/filter_segmentation/sandbox/vincent/grabcut-c++/ref_image.png"
//let path = "/Users/vincent/data/Work/Projects/filter_segmentation/sandbox/vincent/grabcut-c++/ref_pred.png"

let nsImage = NSImage(contentsOfFile: path)
let data = nsImage?.tiffRepresentation
let bitmapRepresentation = NSBitmapImageRep(data: data!)

// Image parameters
let width = Int(nsImage!.size.width)
let height = Int(nsImage!.size.height)

// Create pixel buffer
var pixelBuffer: CVPixelBuffer?
let status = CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_24RGB, nil, &pixelBuffer)
if status != kCVReturnSuccess {
    print("Error: could not create resized pixel buffer", status)
}

// Get buffer
CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags.readOnly)
let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer!)
let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)

// Get bytes per row
let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer!)
let bpp = bytesPerRow / width

// Display
print("Width  = \(CVPixelBufferGetWidth(pixelBuffer!))")
print("Height = \(CVPixelBufferGetHeight(pixelBuffer!))")
print("bytesPerRow = \(bytesPerRow)")
print("bpp = \(bpp)")


for j in 0..<height {
    for i in 0..<width {
        
        let r = UInt8(255 * Float(bitmapRepresentation!.colorAt(x: i, y: j)!.redComponent))
        let g = UInt8(255 * Float(bitmapRepresentation!.colorAt(x: i, y: j)!.greenComponent))
        let b = UInt8(255 * Float(bitmapRepresentation!.colorAt(x: i, y: j)!.blueComponent))
        
        buffer[j * bytesPerRow + i * bpp + 0] = r
        buffer[j * bytesPerRow + i * bpp + 1] = g
        buffer[j * bytesPerRow + i * bpp + 2] = b
    }
}

GrabCut.display(withImage: buffer,
             width: Int32(width),
             height: Int32(height),
             bytesPerRow: Int32(bytesPerRow))

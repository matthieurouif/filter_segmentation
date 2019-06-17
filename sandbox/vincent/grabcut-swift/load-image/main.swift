//
//  main.swift
//  load-image
//
//  Created by Vincent Garcia on 14/06/2019.
//  Copyright © 2019 Vincent Garcia. All rights reserved.
//

import Foundation
import AppKit

extension NSImage {
    func pixelBuffer() -> CVPixelBuffer? {
        let width = self.size.width
        let height = self.size.height
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        
        ///
//        let cfnumPointer = UnsafeMutablePointer<UnsafeRawPointer>.allocate(capacity: 1)
//        let cfnum = CFNumberCreate(kCFAllocatorDefault, .intType, cfnumPointer)
//        let keys: [CFString] = [kCVPixelBufferCGImageCompatibilityKey, kCVPixelBufferCGBitmapContextCompatibilityKey, kCVPixelBufferBytesPerRowAlignmentKey]
//        let values: [CFTypeRef] = [kCFBooleanTrue, kCFBooleanTrue, cfnum!]
//        let keysPointer = UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 1)
//        let valuesPointer =  UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 1)
//        keysPointer.initialize(to: keys)
//        valuesPointer.initialize(to: values)
//        let attrs = CFDictionaryCreate(kCFAllocatorDefault, keysPointer, valuesPointer, keys.count, nil, nil)
//        ///
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(width),
                                         Int(height),
//                                         kCVPixelFormatType_32ARGB,
                                         kCVPixelFormatType_32BGRA,
                                         attrs,
                                         &pixelBuffer)
        
        guard let resultPixelBuffer = pixelBuffer, status == kCVReturnSuccess else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(resultPixelBuffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                      width: Int(width),
                                      height: Int(height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(resultPixelBuffer),
                                      space: rgbColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {return nil}
        
        context.translateBy(x: 0, y: height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        let graphicsContext = NSGraphicsContext(cgContext: context, flipped: false)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = graphicsContext
        draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        NSGraphicsContext.restoreGraphicsState()
        
        CVPixelBufferUnlockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return resultPixelBuffer
    }
}


//func pixelBufferFromImage(image: NSImage) -> CVPixelBuffer {
//
//
//    let ciimage = CIImage(image: image)
//    //let cgimage = convertCIImageToCGImage(inputImage: ciimage!)
//    let tmpcontext = CIContext(options: nil)
//    let cgimage =  tmpcontext.createCGImage(ciimage!, from: ciimage!.extent)
//
//    let cfnumPointer = UnsafeMutablePointer<UnsafeRawPointer>.allocate(capacity: 1)
//    let cfnum = CFNumberCreate(kCFAllocatorDefault, .intType, cfnumPointer)
//    let keys: [CFString] = [kCVPixelBufferCGImageCompatibilityKey, kCVPixelBufferCGBitmapContextCompatibilityKey, kCVPixelBufferBytesPerRowAlignmentKey]
//    let values: [CFTypeRef] = [kCFBooleanTrue, kCFBooleanTrue, cfnum!]
//    let keysPointer = UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 1)
//    let valuesPointer =  UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 1)
//    keysPointer.initialize(to: keys)
//    valuesPointer.initialize(to: values)
//
//    let options = CFDictionaryCreate(kCFAllocatorDefault, keysPointer, valuesPointer, keys.count, nil, nil)
//
//    let width = cgimage!.width
//    let height = cgimage!.height
//
//    var pxbuffer: CVPixelBuffer?
//    // if pxbuffer = nil, you will get status = -6661
//    var status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
//                                     kCVPixelFormatType_32BGRA, options, &pxbuffer)
//    status = CVPixelBufferLockBaseAddress(pxbuffer!, CVPixelBufferLockFlags(rawValue: 0));
//
//    let bufferAddress = CVPixelBufferGetBaseAddress(pxbuffer!);
//
//
//    let rgbColorSpace = CGColorSpaceCreateDeviceRGB();
//    let bytesperrow = CVPixelBufferGetBytesPerRow(pxbuffer!)
//    let context = CGContext(data: bufferAddress,
//                            width: width,
//                            height: height,
//                            bitsPerComponent: 8,
//                            bytesPerRow: bytesperrow,
//                            space: rgbColorSpace,
//                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue);
//    context?.concatenate(CGAffineTransform(rotationAngle: 0))
//    context?.concatenate(__CGAffineTransformMake( 1, 0, 0, -1, 0, CGFloat(height) )) //Flip Vertical
//    //        context?.concatenate(__CGAffineTransformMake( -1.0, 0.0, 0.0, 1.0, CGFloat(width), 0.0)) //Flip Horizontal
//
//
//    context?.draw(cgimage!, in: CGRect(x:0, y:0, width:CGFloat(width), height:CGFloat(height)));
//    status = CVPixelBufferUnlockBaseAddress(pxbuffer!, CVPixelBufferLockFlags(rawValue: 0));
//    return pxbuffer!;
//
//}

//func buffer(from image: NSImage) -> CVPixelBuffer? {
//    let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
//    var pixelBuffer : CVPixelBuffer?
//    let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
//    guard (status == kCVReturnSuccess) else {
//        return nil
//    }
//
//    CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
//    let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
//
//    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
//    let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
//
//    context?.translateBy(x: 0, y: image.size.height)
//    context?.scaleBy(x: 1.0, y: -1.0)
//
//    UIGraphicsPushContext(context!)
//    image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
//    UIGraphicsPopContext()
//    CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
//
//    return pixelBuffer
//}



print(Bundle.main.bundlePath)


//let nsImage = NSImage(contentsOfFile: "/Users/vgarcia/Desktop/dog.jpg")
//let nsImage = NSImage(contentsOfFile: "/Users/vincent/Desktop/50-100-150.jpg")
let nsImage = NSImage(contentsOfFile: "/Users/vincent/Desktop/dog.jpg")
print(nsImage)
//print("Width  = \(nsImage!.size.width)")
//print("Height = \(nsImage!.size.height)")

//let cvImage = nsImage?.pixelBuffer()
////print(cvImage)
//
//let height = CVPixelBufferGetHeight(cvImage!)
//let width = CVPixelBufferGetWidth(cvImage!)
//let bpr = CVPixelBufferGetBytesPerRow(cvImage!)
//let type = CVPixelBufferGetPixelFormatType(cvImage!)


//// Renaud's solution
//let data = nsImage?.tiffRepresentation
//let bitmapRepresentation = NSBitmapImageRep(data: data!)
//let r = Float(bitmapRepresentation!.colorAt(x: 232, y: 547)!.redComponent)
//let g = Float(bitmapRepresentation!.colorAt(x: 232, y: 547)!.greenComponent)
//let b = Float(bitmapRepresentation!.colorAt(x: 232, y: 547)!.blueComponent)
//print(Int(r*255.0))
//print(Int(g*255.0))
//print(Int(b*255.0))
////print(bitmapRepresentation?.colorAt(x: 0, y: 0)?.greenComponent*255)
//// Renaud's solution


// An another solution
let data = nsImage?.tiffRepresentation
let ciImage = CIImage(data: data!)
let pb = ciImage?.pixelBuffer
CVPixelBufferLockBaseAddress(pb!, CVPixelBufferLockFlags.readOnly)
let baseAddress = CVPixelBufferGetBaseAddress(pb!)
let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)

// Get bytes per row
let height = CVPixelBufferGetHeight(pb!)
let width = CVPixelBufferGetWidth(pb!)
let bpr = CVPixelBufferGetBytesPerRow(pb!)
let type = CVPixelBufferGetPixelFormatType(pb!)
let bytesPerRow = CVPixelBufferGetBytesPerRow(pb!)
let bpp = bytesPerRow / width
let bpe = bpr / width

print("Height        = ", height)
print("Width         = ", width)
print("Bytes per row = ", bpr)
print("Bytes per elt = ", bpe)
// An another solution


//uchar c[4];
//c[0] = (type >> 24) & 0xFF;
//c[1] = (type >> 16) & 0xFF;
//c[2] = (type >> 8) & 0xFF;
//c[3] = (type >> 0) & 0xFF;
////NSString *string = [NSString stringWithCharacters:c length:4];

//// Get buffer
//CVPixelBufferLockBaseAddress(cvImage!, CVPixelBufferLockFlags.readOnly)
//let baseAddress = CVPixelBufferGetBaseAddress(cvImage!)
//let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)
//
//// Get bytes per row
//let bytesPerRow = CVPixelBufferGetBytesPerRow(cvImage!)
//let bpp = bytesPerRow / width
//let bpe = bpr / width
//
//print("Height        = ", height)
//print("Width         = ", width)
//print("Bytes per row = ", bpr)
//print("Bytes per elt = ", bpe)
//print("Type          = \(type)")
//
//
//
////let r = buffer[547 * bytesPerRow + 232 * 4 + 0]
////let g = buffer[547 * bytesPerRow + 232 * 4 + 1]
////let b = buffer[547 * bytesPerRow + 232 * 4 + 2]
////print("Swift2 : (\(547), \(232)) = (\(r),\(g),\(b))") // -> 137, 16, 21
//
//
//for i in 0..<bytesPerRow {
//    print(buffer[i])
//}
//
//for j in 0..<height {
//    //    print(buffer[i])
//    //}
//    for i in 0..<width {
//        let r = buffer[j * bytesPerRow + i * bpe + 0]
//        let g = buffer[j * bytesPerRow + i * bpe + 1]
//        let b = buffer[j * bytesPerRow + i * bpe + 2]
//        print("Swift2 : (\(j), \(i)) = (\(r),\(g),\(b))")
//    }
//}

//// Display values
//for j in 0..<height {
//    for i in 0..<width {
//        let r = buffer[j * bytesPerRow + i * 4 + 0]
//        let g = buffer[j * bytesPerRow + i * 4 + 1]
//        let b = buffer[j * bytesPerRow + i * 4 + 2]
//        print("Swift2 : (\(j), \(i)) = (\(r),\(g),\(b))")
//    }
//}

//let imageData = image_ns!.tiffRepresentation
//
//let ciImage = CIImage(data: imageData!)
//print(ciImage)
//
//let pbImage = ciImage?.pixelBuffer
//print(pbImage)

//let path = Bundle.main.path(forResource: "dog", ofType: "jpg")
//print("path = \(path)")
//
//UIMA

//if let path = Bundle.main.path(forResource: "Data", ofType: "json")
//{
//    if let contents = try? String(contentsOfFile: path)
//    {
//        print (contents)
//    }
//    else { print("Could not load contents of file") }
//}
//else { print("Could not find path") }

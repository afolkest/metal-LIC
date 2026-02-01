import AVFoundation
import Metal
import CoreVideo

final class VideoRecorder {
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var startTime: CFAbsoluteTime = 0
    private let queue = DispatchQueue(label: "video-recorder")

    var isRecording: Bool { assetWriter != nil }

    func startRecording(width: Int, height: Int) throws -> URL {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd-HHmmss"
        let timestamp = formatter.string(from: Date())
        let url = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LIC-\(timestamp).mp4")

        let writer = try AVAssetWriter(url: url, fileType: .mp4)

        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ])

        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        self.assetWriter = writer
        self.videoInput = input
        self.adaptor = adaptor
        self.startTime = CFAbsoluteTimeGetCurrent()

        return url
    }

    func appendFrame(from buffer: MTLBuffer, width: Int, height: Int) {
        queue.sync {
            guard let adaptor = adaptor,
                  let input = videoInput,
                  input.isReadyForMoreMediaData else { return }

            var pb: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                kCVPixelFormatType_32BGRA, nil, &pb)
            guard let pixelBuffer = pb else { return }

            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

            let dest = CVPixelBufferGetBaseAddress(pixelBuffer)!
            let destBPR = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let srcBPR = width * 4

            if destBPR == srcBPR {
                memcpy(dest, buffer.contents(), height * srcBPR)
            } else {
                for row in 0..<height {
                    memcpy(dest + row * destBPR,
                           buffer.contents() + row * srcBPR, srcBPR)
                }
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let time = CMTime(seconds: elapsed, preferredTimescale: 600)
            adaptor.append(pixelBuffer, withPresentationTime: time)
        }
    }

    func stopRecording(completion: @escaping (URL?) -> Void) {
        guard let writer = assetWriter else {
            completion(nil)
            return
        }
        let url = writer.outputURL
        videoInput?.markAsFinished()
        writer.finishWriting { completion(url) }
        self.assetWriter = nil
        self.videoInput = nil
        self.adaptor = nil
    }
}

import ArgumentParser
import Foundation
import HuggingFace
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

struct BenchmarkResult: Codable {
    let model: String
    let mode: String
    let contextLabel: String
    let promptTokens: Int
    let generatedTokens: Int
    let timeToFirstTokenMs: Double
    let totalGenerationMs: Double
    let tokensPerSecond: Double
    let peakMemoryMB: Double
    let outputSnippet: String
}

@main
struct TurboQuantBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Benchmark fp16 vs TurboQuant KV cache compression on MLX LLMs",
        version: "0.1.0"
    )

    @Option(name: .long, help: "HuggingFace model ID (e.g. mlx-community/gemma-4-e2b-it-4bit)")
    var model: String = "mlx-community/gemma-4-e2b-it-4bit"

    @Option(name: .long, help: "Number of tokens to generate per run")
    var maxTokens: Int = 256

    @Option(name: .long, help: "Number of warmup runs before measurement")
    var warmup: Int = 1

    @Option(name: .long, help: "Number of measured runs per configuration")
    var runs: Int = 3

    @Option(name: .long, help: "Output JSON file path")
    var output: String = "benchmark_results.json"

    @Flag(name: .long, help: "Run only fp16 (skip TurboQuant)")
    var fp16Only: Bool = false

    @Flag(name: .long, help: "Run only TurboQuant (skip fp16)")
    var turboOnly: Bool = false

    mutating func run() async throws {
        print(String(repeating: "=", count: 60))
        print("TurboQuant KV Cache Benchmark")
        print(String(repeating: "=", count: 60))
        print("Model: \(model)")
        print("Max tokens: \(maxTokens)")
        print("Warmup: \(warmup), Runs: \(runs)")
        print("")

        let config = MLXLMCommon.ModelConfiguration(id: model)

        print("Loading model...")
        let container = try await LLMModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: config
        ) { progress in
            let pct = Int(progress.fractionCompleted * 100)
            if pct % 10 == 0 { print("  Download: \(pct)%") }
        }
        print("Model loaded.\n")

        let workloads: [(label: String, prompt: String)] = [
            ("short-256", makePrompt(targetTokens: 256)),
            ("medium-2k", makePrompt(targetTokens: 2048)),
            ("long-8k", makePrompt(targetTokens: 8192)),
        ]

        var results: [BenchmarkResult] = []

        for (label, prompt) in workloads {
            print(String(repeating: "-", count: 60))
            print("Workload: \(label)")
            print(String(repeating: "-", count: 60))

            if !turboOnly {
                let fp16Results = try await benchmarkMode(
                    container: container, mode: "fp16", turboEnabled: false,
                    prompt: prompt, contextLabel: label
                )
                results.append(contentsOf: fp16Results)
            }

            if !fp16Only {
                let turboResults = try await benchmarkMode(
                    container: container, mode: "turboKV", turboEnabled: true,
                    prompt: prompt, contextLabel: label
                )
                results.append(contentsOf: turboResults)
            }
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(results)
        try data.write(to: URL(fileURLWithPath: output))
        print("\nResults written to \(output)")

        printSummaryTable(results)
    }

    func benchmarkMode(
        container: ModelContainer,
        mode: String,
        turboEnabled: Bool,
        prompt: String,
        contextLabel: String
    ) async throws -> [BenchmarkResult] {
        print("\n  Mode: \(mode)")
        var measured: [BenchmarkResult] = []

        for i in 0..<(warmup + runs) {
            let isWarmup = i < warmup
            let label = isWarmup ? "warmup \(i + 1)" : "run \(i - warmup + 1)"
            print("    \(label)...", terminator: " ")

            let memBefore = currentMemoryMB()
            let startTime = ContinuousClock.now

            var firstTokenTime: ContinuousClock.Instant?
            var tokenCount = 0
            var outputText = ""

            let session = ChatSession(container)
            session.turboQuantEnabled = turboEnabled
            session.generateParameters.maxTokens = maxTokens

            for try await chunk in session.streamResponse(to: prompt) {
                if firstTokenTime == nil {
                    firstTokenTime = ContinuousClock.now
                }
                tokenCount += 1
                outputText += chunk
            }

            let endTime = ContinuousClock.now
            let memAfter = currentMemoryMB()

            let totalMs = Double((endTime - startTime).components.attoseconds) / 1e15
            let ttftMs = firstTokenTime.map {
                Double(($0 - startTime).components.attoseconds) / 1e15
            } ?? 0
            let tps = tokenCount > 0 ? Double(tokenCount) / (totalMs / 1000.0) : 0
            let peakMem = max(memBefore, memAfter)

            let snippet = String(outputText.prefix(80))
                .replacingOccurrences(of: "\n", with: " ")

            print("tokens=\(tokenCount) tok/s=\(String(format: "%.1f", tps)) TTFT=\(String(format: "%.0f", ttftMs))ms mem=\(String(format: "%.0f", peakMem))MB")

            if !isWarmup {
                measured.append(BenchmarkResult(
                    model: model,
                    mode: mode,
                    contextLabel: contextLabel,
                    promptTokens: estimateTokenCount(prompt),
                    generatedTokens: tokenCount,
                    timeToFirstTokenMs: ttftMs,
                    totalGenerationMs: totalMs,
                    tokensPerSecond: tps,
                    peakMemoryMB: peakMem,
                    outputSnippet: snippet
                ))
            }
        }

        return measured
    }

    func makePrompt(targetTokens: Int) -> String {
        let baseContext = """
        The following is a detailed meeting transcript between several team members \
        discussing quarterly planning, product roadmap priorities, and technical \
        architecture decisions. Please analyze this transcript and provide a concise \
        summary with action items.

        """
        if targetTokens <= 512 {
            return baseContext + "What are the key priorities for Q3?"
        }

        var text = baseContext
        let fillerSentences = [
            "Sarah mentioned that the API latency has been increasing over the past sprint. ",
            "Tom suggested we prioritize the database migration before adding new features. ",
            "The team agreed to schedule a design review for the new onboarding flow next week. ",
            "Marketing requested early access to the beta for press materials by end of month. ",
            "Engineering capacity is constrained due to two team members being on vacation. ",
            "The CI pipeline improvements should reduce build times by approximately 40%. ",
            "We need to evaluate whether to continue with the current caching strategy. ",
            "Customer feedback indicates strong demand for offline mode functionality. ",
            "The security audit findings require immediate attention on authentication flows. ",
            "Performance testing on M-series Macs shows significant improvements in inference. ",
        ]

        let wordsPerToken = 1.3
        let targetWords = Int(Double(targetTokens) * wordsPerToken)
        var wordCount = text.split(separator: " ").count

        while wordCount < targetWords {
            let sentence = fillerSentences[wordCount % fillerSentences.count]
            text += sentence
            wordCount += sentence.split(separator: " ").count
        }

        text += "\n\nPlease summarize the key discussion points, decisions made, and action items."
        return text
    }

    func estimateTokenCount(_ text: String) -> Int {
        Int(Double(text.split(separator: " ").count) / 1.3)
    }

    func currentMemoryMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            return Double(info.resident_size) / 1_048_576
        }
        return 0
    }

    func printSummaryTable(_ results: [BenchmarkResult]) {
        print("\n" + String(repeating: "=", count: 80))
        print("BENCHMARK SUMMARY")
        print(String(repeating: "=", count: 80))
        print(String(format: "%-12s %-10s %8s %8s %8s %10s",
                      "Context", "Mode", "Tok/s", "TTFT(ms)", "Gen(ms)", "Mem(MB)"))
        print(String(repeating: "-", count: 80))

        let grouped = Dictionary(grouping: results) { "\($0.contextLabel)|\($0.mode)" }
        let sortedKeys = grouped.keys.sorted()

        for key in sortedKeys {
            guard let group = grouped[key], !group.isEmpty else { continue }
            let avgTps = group.map(\.tokensPerSecond).reduce(0, +) / Double(group.count)
            let avgTtft = group.map(\.timeToFirstTokenMs).reduce(0, +) / Double(group.count)
            let avgGen = group.map(\.totalGenerationMs).reduce(0, +) / Double(group.count)
            let avgMem = group.map(\.peakMemoryMB).reduce(0, +) / Double(group.count)

            print(String(format: "%-12s %-10s %8.1f %8.0f %8.0f %10.0f",
                          group[0].contextLabel, group[0].mode,
                          avgTps, avgTtft, avgGen, avgMem))
        }

        print(String(repeating: "=", count: 80))
        print("Model: \(results.first?.model ?? "unknown")")
        print("Runs per config: \(runs)")
        let hw = ProcessInfo.processInfo
        print("Hardware: \(hw.processorCount) cores / \(hw.physicalMemory / (1024*1024*1024)) GB RAM")
        print("")
    }
}

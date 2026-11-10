import AppKit
import SwiftUI
import TipKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let model = AppModel()

    func applicationWillFinishLaunching(_ notification: Notification) {
        // A quit from the Dock, a logout, restart or shut down arrives as this Apple Event, which AppKit
        // refuses while a sheet is open (the first-run sheet opens at launch). It takes the Quit command's
        // path instead: ask during a download, close any sheet, then quit (task 19).
        NSAppleEventManager.shared().setEventHandler(
            self, andSelector: #selector(handleQuit(_:withReplyEvent:)),
            forEventClass: AEEventClass(kCoreEventClass), andEventID: AEEventID(kAEQuitApplication))
    }

    @objc private func handleQuit(_ event: NSAppleEventDescriptor, withReplyEvent reply: NSAppleEventDescriptor) {
        model.quit()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        AppAppearance.stored.apply()
        NSApp.activate()
        DevHarness.startIfRequested(model)
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        return model.confirmQuit() ? .terminateNow : .terminateCancel
    }

    func applicationWillTerminate(_ notification: Notification) {
        model.shutdown()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct DMLXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    private var model: AppModel { delegate.model }

    init() { Guide.configure() }

    var body: some Scene {
        Window("dMLX", id: "main") {
            RootView()
                .environment(model)
                // Stay usable on compact or scaled secondary displays. The
                // split view and composer adapt when the detail column narrows.
                .frame(minWidth: 680, minHeight: 520)
                .tint(Theme.accent)
        }
        .windowToolbarStyle(.unified(showsTitle: false))
        .commands {
            // AppKit ignores Quit while a sheet is open (the first-run sheet opens at launch), so Quit
            // closes any sheet first, then quits (task 19). The Dock's Quit and a logout come in as an
            // Apple Event and take the same path (AppDelegate).
            CommandGroup(replacing: .appTermination) {
                Button("Quit dMLX") { model.quit() }.keyboardShortcut("q")
            }
            CommandGroup(replacing: .newItem) {
                Button("New Creation") { model.newSession() }.keyboardShortcut("n")
            }
            CommandGroup(replacing: .help) {
                Link("dMLX Guide", destination: URL(string: "https://dmlx.vercel.app/docs/using-dmlx/")!)
                Button("Show Tips Again") { Guide.shared.showAgain() }
            }
            // In the app menu under Settings…: setup opens by itself only on a first launch, and this brings
            // back the image model and agent choice.
            CommandGroup(after: .appSettings) {
                Button("Choose Models…") {
                    model.onboardingStep = .setup
                    model.showOnboarding = true
                }
                Divider()
                Button("Uninstall dMLX…") { model.uninstall() }
            }
        }

        Settings {
            SettingsView()
                .environment(model)
                .frame(width: 620, height: 460)
                .tint(Theme.accent)
        }
    }
}

/// Settings → General → Appearance. "System" follows macOS.
enum AppAppearance: String, CaseIterable, Identifiable {
    case system, light, dark
    static let key = "appearance"
    var id: String { rawValue }
    var title: String {
        switch self {
        case .system: "System"
        case .light: "Light"
        case .dark: "Dark"
        }
    }
    static var stored: AppAppearance {
        AppAppearance(rawValue: UserDefaults.standard.string(forKey: key) ?? "") ?? .system
    }
    @MainActor func apply() {
        switch self {
        case .system: NSApp.appearance = nil
        case .light: NSApp.appearance = NSAppearance(named: .aqua)
        case .dark: NSApp.appearance = NSAppearance(named: .darkAqua)
        }
    }
}

enum Theme {
    /// Brand colors from docs/brand/README.md ("Lake"). Each has a light and a dark version.
    private static func adaptive(light: UInt32, dark: UInt32) -> Color {
        func ns(_ hex: UInt32) -> NSColor {
            NSColor(srgbRed: CGFloat(hex >> 16 & 0xff) / 255, green: CGFloat(hex >> 8 & 0xff) / 255,
                    blue: CGFloat(hex & 0xff) / 255, alpha: 1)
        }
        return Color(nsColor: NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? ns(dark) : ns(light)
        })
    }
    /// The sun in the logo.
    static let logo = adaptive(light: 0x2f6fa3, dark: 0x7db7e8)
    /// Buttons, links and selection. In dark mode it is deeper than the logo blue so that white
    /// text on a prominent button stays readable.
    static let accent = adaptive(light: 0x2f6fa3, dark: 0x3a84cf)
    static let corner: CGFloat = 12
}

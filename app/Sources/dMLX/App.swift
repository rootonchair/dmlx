import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let model = AppModel()

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate()
        DevHarness.startIfRequested(model)
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

    var body: some Scene {
        Window("dMLX", id: "main") {
            RootView()
                .environment(model)
                .frame(minWidth: 820, minHeight: 600)
                .tint(Theme.accent)
        }
        .windowToolbarStyle(.unified(showsTitle: false))
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Creation") { model.newSession() }.keyboardShortcut("n")
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

enum Theme {
    static let accent = Color(red: 0.91, green: 0.64, blue: 0.24)
    static let corner: CGFloat = 12
}

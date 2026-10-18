import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    var model: AppModel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Launched from `swift run` there is no bundle; make sure we still get a Dock icon and focus.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate()
    }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { model?.shutdown() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct DMLXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup("dMLX") {
            RootView()
                .environment(model)
                .frame(minWidth: 820, minHeight: 600)
                .onAppear { delegate.model = model }
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

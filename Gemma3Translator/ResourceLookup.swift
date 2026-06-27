import Foundation

enum ResourceLookup {
    static func path(_ name: String, ext: String) -> String? {
        if let p = Bundle.main.path(forResource: name, ofType: ext) { return p }
        return Bundle.main.path(forResource: name, ofType: ext, inDirectory: "Resources")
    }
    static func url(_ name: String, ext: String) -> URL? {
        if let u = Bundle.main.url(forResource: name, withExtension: ext) { return u }
        return Bundle.main.url(forResource: name, withExtension: ext, subdirectory: "Resources")
    }
}

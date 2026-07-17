#include "model_config.h"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace tts {
namespace {

class JsonReader {
public:
    explicit JsonReader(const std::string& input) : input_(input) {}

    bool parse(ModelConfig& config, std::string& error) {
        try {
            parse_root(config);
            skip_whitespace();
            if (position_ != input_.size()) fail("unexpected trailing content");
            return true;
        } catch (const std::runtime_error& ex) {
            error = ex.what();
            return false;
        }
    }

private:
    const std::string& input_;
    size_t position_ = 0;

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("JSON error at byte " + std::to_string(position_) + ": " + message);
    }

    void skip_whitespace() {
        while (position_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
    }

    bool consume(char expected) {
        skip_whitespace();
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) fail(std::string("expected '") + expected + "'");
    }

    static void append_utf8(std::string& output, unsigned codepoint) {
        if (codepoint <= 0x7F) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FF) {
            output.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else if (codepoint <= 0xFFFF) {
            output.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else {
            output.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
    }

    unsigned parse_hex4() {
        if (position_ + 4 > input_.size()) fail("incomplete unicode escape");
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = input_[position_++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<unsigned>(c - 'A' + 10);
            else fail("invalid unicode escape");
        }
        return value;
    }

    std::string parse_string() {
        skip_whitespace();
        if (position_ >= input_.size() || input_[position_++] != '"') fail("expected string");
        std::string value;
        while (position_ < input_.size()) {
            const char c = input_[position_++];
            if (c == '"') return value;
            if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
            if (c != '\\') {
                value.push_back(c);
                continue;
            }
            if (position_ >= input_.size()) fail("incomplete string escape");
            const char escaped = input_[position_++];
            switch (escaped) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case 'u': {
                    unsigned codepoint = parse_hex4();
                    if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                        if (position_ + 2 > input_.size() ||
                            input_[position_] != '\\' || input_[position_ + 1] != 'u') {
                            fail("high surrogate must be followed by a low surrogate");
                        }
                        position_ += 2;
                        const unsigned low = parse_hex4();
                        if (low < 0xDC00 || low > 0xDFFF) fail("invalid low surrogate");
                        codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
                    } else if (codepoint >= 0xDC00 && codepoint <= 0xDFFF) {
                        fail("unexpected low surrogate");
                    }
                    append_utf8(value, codepoint);
                    break;
                }
                default: fail("invalid string escape");
            }
        }
        fail("unterminated string");
    }

    double parse_number() {
        skip_whitespace();
        const size_t begin_position = position_;
        if (position_ < input_.size() && input_[position_] == '-') ++position_;
        if (position_ >= input_.size()) fail("invalid number");

        if (input_[position_] == '0') {
            ++position_;
            if (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
                fail("leading zero in number");
            }
        } else if (input_[position_] >= '1' && input_[position_] <= '9') {
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
        } else {
            fail("invalid number");
        }

        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            const size_t fraction_start = position_;
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
            if (position_ == fraction_start) fail("missing fraction digits");
        }

        if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            const size_t exponent_start = position_;
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
            if (position_ == exponent_start) fail("missing exponent digits");
        }

        const std::string token = input_.substr(begin_position, position_ - begin_position);
        const char* begin = token.c_str();
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(begin, &end);
        if (end == begin || *end != '\0' || errno == ERANGE || !std::isfinite(value)) fail("invalid number");
        return value;
    }

    void parse_literal(const char* literal) {
        while (*literal) {
            if (position_ >= input_.size() || input_[position_++] != *literal++) fail("invalid literal");
        }
    }

    void skip_value() {
        skip_whitespace();
        if (position_ >= input_.size()) fail("expected value");
        if (input_[position_] == '"') {
            parse_string();
        } else if (input_[position_] == '{') {
            expect('{');
            if (consume('}')) return;
            do {
                parse_string();
                expect(':');
                skip_value();
            } while (consume(','));
            expect('}');
        } else if (input_[position_] == '[') {
            expect('[');
            if (consume(']')) return;
            do {
                skip_value();
            } while (consume(','));
            expect(']');
        } else if (input_[position_] == 't') {
            parse_literal("true");
        } else if (input_[position_] == 'f') {
            parse_literal("false");
        } else if (input_[position_] == 'n') {
            parse_literal("null");
        } else {
            parse_number();
        }
    }

    void parse_models(ModelConfig& config) {
        expect('{');
        if (consume('}')) return;
        do {
            const std::string key = parse_string();
            expect(':');
            const std::string value = parse_string();
            if (key.empty() || value.empty()) fail("model keys and paths must not be empty");
            if (!config.models.emplace(key, value).second) fail("duplicate model key '" + key + "'");
        } while (consume(','));
        expect('}');
    }

    AdapterConfig parse_adapter() {
        AdapterConfig adapter;
        std::unordered_set<std::string> fields;
        expect('{');
        if (!consume('}')) {
            do {
                const std::string key = parse_string();
                if (!fields.emplace(key).second) fail("duplicate adapter field '" + key + "'");
                expect(':');
                if (key == "name") adapter.name = parse_string();
                else if (key == "path") adapter.path = parse_string();
                else if (key == "target") adapter.target = parse_string();
                else if (key == "scale") adapter.scale = static_cast<float>(parse_number());
                else skip_value();
            } while (consume(','));
            expect('}');
        }
        if (adapter.path.empty()) fail("adapter path must not be empty");
        if (!std::isfinite(adapter.scale)) fail("adapter scale must be finite");
        return adapter;
    }

    void parse_adapters(ModelConfig& config) {
        expect('[');
        if (consume(']')) return;
        do {
            config.adapters.push_back(parse_adapter());
        } while (consume(','));
        expect(']');
    }

    void parse_root(ModelConfig& config) {
        std::string format;
        std::unordered_set<std::string> fields;
        expect('{');
        if (!consume('}')) {
            do {
                const std::string key = parse_string();
                if (!fields.emplace(key).second) fail("duplicate root field '" + key + "'");
                expect(':');
                if (key == "format") format = parse_string();
                else if (key == "name") config.name = parse_string();
                else if (key == "provider") config.provider = parse_string();
                else if (key == "models") parse_models(config);
                else if (key == "adapters") parse_adapters(config);
                else skip_value();
            } while (consume(','));
            expect('}');
        }

        if (format != "tts-model") fail("format must be 'tts-model'");
        if (config.name.empty()) fail("name must not be empty");
        if (config.provider.empty()) fail("provider must not be empty");
        if (config.models.empty()) fail("models must contain at least one entry");
    }
};

} // namespace

std::filesystem::path ModelConfig::resolve_path(const std::string& value) const {
    const std::filesystem::path path = std::filesystem::u8path(value);
    return path.is_absolute() ? path.lexically_normal() : (base_directory / path).lexically_normal();
}

bool load_model_config(const std::filesystem::path& path, ModelConfig& config, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "failed to open model config: " + path.string();
        return false;
    }

    std::ostringstream contents;
    contents << input.rdbuf();
    ModelConfig parsed;
    const std::string json = contents.str();
    JsonReader reader(json);
    if (!reader.parse(parsed, error)) return false;

    parsed.source_path = std::filesystem::absolute(path).lexically_normal();
    parsed.base_directory = parsed.source_path.parent_path();
    config = std::move(parsed);
    return true;
}

} // namespace tts

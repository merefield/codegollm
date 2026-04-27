package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCleanPathRejectsEscapes(t *testing.T) {
	root := t.TempDir()
	tests := []struct {
		name    string
		path    string
		want    string
		wantErr bool
	}{
		{name: "relative", path: "src/main.go", want: filepath.Join("src", "main.go")},
		{name: "dot segment", path: "src/../README.md", want: "README.md"},
		{name: "parent escape", path: "../secret", wantErr: true},
		{name: "absolute", path: filepath.Join(root, "file"), wantErr: true},
		{name: "empty", path: "", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := cleanPath(root, tt.path)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got path %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestParseEnvLine(t *testing.T) {
	tests := []struct {
		line    string
		key     string
		value   string
		ok      bool
		wantErr bool
	}{
		{line: "OPENAI_API_KEY=sk-test", key: "OPENAI_API_KEY", value: "sk-test", ok: true},
		{line: "export TOKEN='abc 123'", key: "TOKEN", value: "abc 123", ok: true},
		{line: "NAME=value # comment", key: "NAME", value: "value", ok: true},
		{line: "# comment", ok: false},
		{line: "bad-line", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.line, func(t *testing.T) {
			key, value, ok, err := parseEnvLine(tt.line)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if ok != tt.ok || key != tt.key || value != tt.value {
				t.Fatalf("got key=%q value=%q ok=%v", key, value, ok)
			}
		})
	}
}

func TestLoadEnvFileDoesNotOverwriteExisting(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".env")
	if err := os.WriteFile(path, []byte("CODEGOLLM_TEST_ENV=from-file\nCODEGOLLM_TEST_NEW=loaded\n"), 0644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CODEGOLLM_TEST_ENV", "from-shell")
	t.Setenv("CODEGOLLM_TEST_NEW", "")
	if err := os.Unsetenv("CODEGOLLM_TEST_NEW"); err != nil {
		t.Fatal(err)
	}

	loaded, err := loadEnvFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded != 1 {
		t.Fatalf("loaded %d vars, want 1", loaded)
	}
	if got := os.Getenv("CODEGOLLM_TEST_ENV"); got != "from-shell" {
		t.Fatalf("existing env overwritten: %q", got)
	}
	if got := os.Getenv("CODEGOLLM_TEST_NEW"); got != "loaded" {
		t.Fatalf("new env not loaded: %q", got)
	}
}

func TestLoadConfigDefaultsAndLegacyOllamaInference(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		dir := t.TempDir()
		oldWd, err := os.Getwd()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.Chdir(oldWd) })
		if err := os.Chdir(dir); err != nil {
			t.Fatal(err)
		}

		cfg, _, err := loadConfig()
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Provider != "openai" || cfg.Model != "gpt-4.1-mini" || cfg.OpenAIAPIKeyEnv != "OPENAI_API_KEY" {
			t.Fatalf("unexpected defaults: %+v", cfg)
		}
	})

	t.Run("legacy ollama model", func(t *testing.T) {
		dir := t.TempDir()
		oldWd, err := os.Getwd()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.Chdir(oldWd) })
		if err := os.Chdir(dir); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile("config.yaml", []byte("model: gemma4:latest\n"), 0644); err != nil {
			t.Fatal(err)
		}

		cfg, _, err := loadConfig()
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Provider != "ollama" {
			t.Fatalf("provider = %q, want ollama", cfg.Provider)
		}
	})
}

func TestParseToolArgsAcceptsNestedAndStringifiedJSON(t *testing.T) {
	nested := json.RawMessage(`{"input":{"file-path":"script.js","old_text":"a","new_text":"b"}}`)
	args, err := parseToolArgs(nested)
	if err != nil {
		t.Fatal(err)
	}
	if argValue(args, "file_path") != "script.js" ||
		argValue(args, "old", "old_text") != "a" ||
		argValue(args, "new", "new_text") != "b" {
		t.Fatalf("unexpected nested args: %#v", args)
	}

	stringified := json.RawMessage(`"{\"path\":\"main.go\",\"content\":\"package main\"}"`)
	args, err = parseToolArgs(stringified)
	if err != nil {
		t.Fatal(err)
	}
	if args["path"] != "main.go" || args["content"] != "package main" {
		t.Fatalf("unexpected stringified args: %#v", args)
	}
}

func TestPrepareToolAliasesAndPreview(t *testing.T) {
	root := t.TempDir()
	m := model{root: root}
	call := ToolCall{Function: ToolFunction{
		Name:      "write",
		Arguments: json.RawMessage(`{"file_path":"script.js","text":"console.log(1);\n"}`),
	}}
	req, err := m.prepareTool(call)
	if err != nil {
		t.Fatal(err)
	}
	if req.Args["path"] != "script.js" || req.Args["content"] != "console.log(1);" {
		t.Fatalf("unexpected request args: %#v", req.Args)
	}
	if !strings.Contains(req.Diff, "+console.log(1);") {
		t.Fatalf("diff did not include added content: %q", req.Diff)
	}
}

func TestApprovalKeyAndHasApproval(t *testing.T) {
	req := toolRequest{Name: "bash", Args: map[string]string{"command": "  deno lint script.js  "}}
	key := approvalKey(req)
	if key != "bash:deno lint script.js" {
		t.Fatalf("key = %q", key)
	}
	if !hasApproval([]string{key}, "bash:deno lint script.js") {
		t.Fatal("expected approval match")
	}
}

func TestContextMessagesUsesSummaryAndRecentWindow(t *testing.T) {
	m := model{
		cfg: Config{SystemPrompt: "system", RecentHistoryMessages: 2},
		messages: []ChatMessage{
			{Role: "system", Content: "system"},
			{Role: "user", Content: "old"},
			{Role: "assistant", Content: "middle"},
			{Role: "user", Content: "new"},
		},
		summary:        "summary",
		summaryThrough: 2,
	}
	context := m.contextMessages()
	if len(context) != 4 {
		t.Fatalf("len(context) = %d, want 4: %#v", len(context), context)
	}
	if context[0].Content != "system" || !strings.Contains(context[1].Content, "summary") {
		t.Fatalf("missing system or summary messages: %#v", context)
	}
	if context[2].Content != "middle" || context[3].Content != "new" {
		t.Fatalf("unexpected recent messages: %#v", context)
	}
}

func TestOpenAIMessagesNormalizesToolHistory(t *testing.T) {
	messages := []ChatMessage{
		{Role: "assistant", ToolCalls: []ToolCall{{Function: ToolFunction{Name: "read", Arguments: json.RawMessage(`{"path":"main.go"}`)}}}},
		{Role: "tool", ToolName: "read", Content: "content without id"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_1", Function: ToolFunction{Name: "bash", Arguments: json.RawMessage(`{"command":"ls"}`)}}}},
	}
	got := openAIMessages(messages)
	if got[0].Role != "assistant" || !strings.Contains(got[0].Content, "assistant tool call") || len(got[0].ToolCalls) != 0 {
		t.Fatalf("assistant without tool id not normalized: %#v", got[0])
	}
	if got[1].Role != "user" || !strings.Contains(got[1].Content, "Prior tool result") {
		t.Fatalf("tool without call id not converted: %#v", got[1])
	}
	if got[2].Content != "" || got[2].ToolCalls[0].Type != "function" {
		t.Fatalf("valid assistant tool call not normalized correctly: %#v", got[2])
	}
	if string(got[2].ToolCalls[0].Function.Arguments)[:1] != `"` {
		t.Fatalf("OpenAI tool arguments should be JSON string, got %s", got[2].ToolCalls[0].Function.Arguments)
	}
}

func TestSessionRoundTripAndSystemPromptRefresh(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".codegollm", "session.json")
	state := SessionState{
		Messages:       []ChatMessage{{Role: "system", Content: "old"}, {Role: "user", Content: "hi"}},
		Summary:        "old work",
		SummaryThrough: 1,
	}
	if err := saveSession(path, state); err != nil {
		t.Fatal(err)
	}
	loaded, err := loadSession(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Summary != "old work" || len(loaded.Messages) != 2 {
		t.Fatalf("bad loaded session: %#v", loaded)
	}
	refreshed := ensureSystemMessage(loaded.Messages, "new system")
	if refreshed[0].Content != "new system" || refreshed[1].Content != "hi" {
		t.Fatalf("system prompt not refreshed: %#v", refreshed)
	}
}

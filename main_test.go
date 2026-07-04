package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
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
		if cfg.AuthProfile != defaultAuthProfile || cfg.ChatGPTIssuer != defaultChatGPTIssuer || cfg.ChatGPTOAuthClientID != defaultChatGPTOAuthClient {
			t.Fatalf("unexpected auth defaults: %+v", cfg)
		}
		if len(cfg.PreferredModels) == 0 || cfg.PreferredModels[0] != "gpt-5.5" {
			t.Fatalf("unexpected preferred model defaults: %+v", cfg.PreferredModels)
		}
		if cfg.ReasoningLevel != "medium" || cfg.Fast {
			t.Fatalf("unexpected reasoning/fast defaults: %+v", cfg)
		}
		if cfg.IncludeReasoningInContext {
			t.Fatalf("include reasoning default = true, want false")
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

func TestCLIOverrides(t *testing.T) {
	overrides, err := parseCLIOverrides([]string{"--reasoning", "high", "--fast"})
	if err != nil {
		t.Fatal(err)
	}
	cfg := applyCLIOverrides(Config{ReasoningLevel: "medium", Fast: false}, overrides)
	if cfg.ReasoningLevel != "high" || !cfg.Fast {
		t.Fatalf("bad overrides: %+v", cfg)
	}

	overrides, err = parseCLIOverrides([]string{"--no-fast"})
	if err != nil {
		t.Fatal(err)
	}
	cfg = applyCLIOverrides(Config{ReasoningLevel: "medium", Fast: true}, overrides)
	if cfg.Fast {
		t.Fatalf("no-fast did not apply: %+v", cfg)
	}

	overrides, err = parseCLIOverrides([]string{"--reasoning", "minimal"})
	if err != nil {
		t.Fatal(err)
	}
	cfg = applyCLIOverrides(Config{ReasoningLevel: "medium"}, overrides)
	if cfg.ReasoningLevel != "minimal" {
		t.Fatalf("minimal reasoning did not apply: %+v", cfg)
	}

	if _, err := parseCLIOverrides([]string{"--reasoning", "maximum"}); err == nil {
		t.Fatal("expected invalid reasoning error")
	}
}

func TestAuthProfilesRoundTripAndResolveEnvKey(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("CODEGOLLM_TEST_OPENAI_KEY", "sk-test")

	if err := registerEnvAPIKeyProfile("test-profile", "CODEGOLLM_TEST_OPENAI_KEY"); err != nil {
		t.Fatal(err)
	}
	store, err := loadAuthProfiles()
	if err != nil {
		t.Fatal(err)
	}
	profile, ok := store.Profiles["test-profile"]
	if !ok {
		t.Fatalf("missing registered profile: %#v", store.Profiles)
	}
	if profile.AuthType != "api_key_env" || profile.Env != "CODEGOLLM_TEST_OPENAI_KEY" {
		t.Fatalf("unexpected profile: %#v", profile)
	}

	auth, err := resolveAuth(context.Background(), Config{Provider: "openai", AuthProfile: "test-profile", OpenAIAPIKeyEnv: "OPENAI_API_KEY"})
	if err != nil {
		t.Fatal(err)
	}
	if auth.APIKey != "sk-test" || auth.Provider != "openai" {
		t.Fatalf("bad resolved auth: %#v", auth)
	}

	path, err := authProfilesPath()
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0600 {
		t.Fatalf("auth store mode = %v, want 0600", info.Mode().Perm())
	}

	if err := deleteAuthProfile("test-profile"); err != nil {
		t.Fatal(err)
	}
	store, err = loadAuthProfiles()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := store.Profiles["test-profile"]; ok {
		t.Fatalf("profile was not deleted: %#v", store.Profiles)
	}
}

func TestJWTClaimsAndAuthorizeURL(t *testing.T) {
	jwt := fakeJWT(t, map[string]any{
		"exp":                float64(time.Unix(1893456000, 0).Unix()),
		"chatgpt_account_id": "acct_123",
	})
	if got := jwtStringClaim(jwt, "chatgpt_account_id"); got != "acct_123" {
		t.Fatalf("account claim = %q", got)
	}
	if got := jwtExpiration(jwt); !got.Equal(time.Unix(1893456000, 0).UTC()) {
		t.Fatalf("expiration = %v", got)
	}

	authURL := chatGPTAuthorizeURL(Config{
		ChatGPTIssuer:        defaultChatGPTIssuer,
		ChatGPTOAuthClientID: defaultChatGPTOAuthClient,
		ChatGPTWorkspaceID:   "org-test",
	}, "http://localhost:1455/auth/callback", "state", pkceCodes{Challenge: "challenge"})
	for _, want := range []string{
		"https://auth.openai.com/oauth/authorize?",
		"client_id=" + defaultChatGPTOAuthClient,
		"redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback",
		"code_challenge=challenge",
		"state=state",
		"originator=codegollm",
		"allowed_workspace_id=org-test",
	} {
		if !strings.Contains(authURL, want) {
			t.Fatalf("authorize URL missing %q: %s", want, authURL)
		}
	}
	if strings.Contains(authURL, "model.request") {
		t.Fatalf("authorize URL requested unsupported model.request scope: %s", authURL)
	}
}

func TestChatGPTOrganizationsAndWorkspaceHint(t *testing.T) {
	idToken := fakeJWT(t, map[string]any{
		"https://api.openai.com/auth": map[string]any{
			"organizations": []any{
				map[string]any{"id": "org-one", "title": "One", "is_default": true},
				map[string]any{"id": "org-two", "title": "Two"},
			},
		},
	})
	orgs := chatGPTOrganizations(idToken)
	if len(orgs) != 2 || orgs[0].ID != "org-one" || !orgs[0].IsDefault || orgs[1].Title != "Two" {
		t.Fatalf("organizations = %#v", orgs)
	}
	hint := chatGPTWorkspaceHint("openai-chatgpt:default", idToken)
	for _, want := range []string{
		"/login openai-chatgpt openai-chatgpt:default org-one",
		"/login openai-chatgpt openai-chatgpt:default org-two",
	} {
		if !strings.Contains(hint, want) {
			t.Fatalf("hint missing %q: %s", want, hint)
		}
	}
}

func fakeJWT(t *testing.T, claims map[string]any) string {
	t.Helper()
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatal(err)
	}
	return "header." + base64.RawURLEncoding.EncodeToString(payload) + ".sig"
}

func TestRecommendedModelUsesPreferredThenHeuristic(t *testing.T) {
	cfg := Config{PreferredModels: []string{"gpt-5.4", "gpt-4.1-mini"}}
	models := []string{"gpt-4.1-mini", "gpt-5", "gpt-5.4"}
	if got := recommendedModel(cfg, models); got != "gpt-5.4" {
		t.Fatalf("recommended model = %q", got)
	}

	cfg.PreferredModels = []string{"not-available"}
	models = []string{"gpt-4.1-mini", "gpt-5-codex", "gpt-5"}
	if got := recommendedModel(cfg, models); got != "gpt-5-codex" {
		t.Fatalf("heuristic recommended model = %q", got)
	}

	cfg = Config{Fast: true, PreferredModels: []string{"gpt-5.5", "gpt-5.4-mini", "gpt-4.1-mini"}}
	models = []string{"gpt-5.5", "gpt-5.4-mini", "gpt-4.1-mini"}
	if got := recommendedModel(cfg, models); got != "gpt-5.4-mini" {
		t.Fatalf("fast recommended model = %q", got)
	}
}

func TestReasoningEffortForRequest(t *testing.T) {
	cfg := Config{ReasoningLevel: "high"}
	if got := reasoningEffortForRequest(cfg, "gpt-5.4"); got != "high" {
		t.Fatalf("reasoning effort = %q", got)
	}
	if got := reasoningEffortForRequest(cfg, "gpt-4.1-mini"); got != "" {
		t.Fatalf("non-reasoning model should omit effort, got %q", got)
	}
	if got := reasoningEffortForRequest(Config{}, "gpt-5-codex"); got != "medium" {
		t.Fatalf("default reasoning effort = %q", got)
	}
	if got := reasoningEffortForRequest(Config{ReasoningLevel: "none"}, "gpt-5"); got != "" {
		t.Fatalf("none should be omitted for pre-5.1 models, got %q", got)
	}
	if got := reasoningEffortForRequest(Config{ReasoningLevel: "none"}, "gpt-5.4"); got != "none" {
		t.Fatalf("none should be allowed for newer models, got %q", got)
	}
	if got := reasoningEffortForRequest(Config{ReasoningLevel: "xhigh"}, "gpt-5"); got != "high" {
		t.Fatalf("xhigh should downgrade for unsupported models, got %q", got)
	}
}

func TestOpenAIModelOrderingAndFiltering(t *testing.T) {
	cfg := Config{PreferredModels: []string{"gpt-5", "gpt-4.1-mini"}}
	models := []string{
		"text-embedding-3-small",
		"gpt-4.1-mini",
		"gpt-realtime",
		"gpt-5",
		"gpt-image-1",
		"gpt-5-codex",
	}
	var filtered []string
	for _, model := range models {
		if isUsefulCodingModel(model) {
			filtered = append(filtered, model)
		}
	}
	if strings.Join(filtered, ",") != "gpt-4.1-mini,gpt-5,gpt-5-codex" {
		t.Fatalf("bad filtered models: %#v", filtered)
	}
	recommended := recommendedModel(cfg, filtered)
	ordered := orderOpenAIModels(cfg, filtered, recommended)
	if strings.Join(ordered, ",") != "gpt-5,gpt-4.1-mini,gpt-5-codex" {
		t.Fatalf("bad ordered models: %#v", ordered)
	}
}

func TestCurateModelChoicesKeepsImportantModelsAndCapsList(t *testing.T) {
	cfg := Config{
		Model:           "gpt-4.1-mini",
		PreferredModels: []string{"gpt-5.4", "gpt-5.4-mini", "gpt-5", "gpt-4.1-mini"},
	}
	models := []string{
		"gpt-5.4",
		"gpt-5.4-mini",
		"gpt-5",
		"gpt-4.1-mini",
		"gpt-extra-1",
		"gpt-extra-2",
		"gpt-extra-3",
		"gpt-extra-4",
	}
	got := curateModelChoices(cfg, models, "gpt-5.4", 5)
	if len(got) != 5 {
		t.Fatalf("len = %d, want 5: %#v", len(got), got)
	}
	for _, want := range []string{"gpt-4.1-mini", "gpt-5.4", "gpt-5.4-mini", "gpt-5"} {
		if !containsString(got, want) {
			t.Fatalf("curated list missing %q: %#v", want, got)
		}
	}
}

func TestModelPickerScrollsCursorIntoView(t *testing.T) {
	m := model{
		height:           20,
		modelCursor:      11,
		modelRecommended: "model-1",
		modelTotal:       20,
		modelChoices: []string{
			"model-0", "model-1", "model-2", "model-3", "model-4",
			"model-5", "model-6", "model-7", "model-8", "model-9",
			"model-10", "model-11", "model-12",
		},
	}
	m = m.ensureModelCursorVisible()
	start, end := m.modelVisibleRange()
	if start != 6 || end != 12 {
		t.Fatalf("visible range = %d:%d, want 6:12", start, end)
	}
	view := m.View()
	if !strings.Contains(view, "model-11") || strings.Contains(view, "model-0") {
		t.Fatalf("picker view did not scroll as expected:\n%s", view)
	}
}

func TestListOpenAIModelsUsesAuthAndFilters(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	if err := saveAuthProfiles(AuthProfiles{Profiles: map[string]AuthProfile{
		"test": {Provider: "openai-compat", AuthType: "api_key", APIKey: "sk-test"},
	}}); err != nil {
		t.Fatal(err)
	}
	oldClient := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = oldClient })
	http.DefaultClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.String() != "https://example.test/models" {
			t.Fatalf("url = %q, want https://example.test/models", r.URL.String())
		}
		if got := r.Header.Get("Authorization"); got != "Bearer sk-test" {
			t.Fatalf("Authorization = %q", got)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{"data":[{"id":"gpt-image-1"},{"id":"gpt-4.1-mini"},{"id":"gpt-5"}]}`)),
		}, nil
	})}

	models, err := listOpenAIModels(context.Background(), Config{
		Provider:      "openai-compat",
		AuthProfile:   "test",
		OpenAIBaseURL: "https://example.test",
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(models, ",") != "gpt-4.1-mini,gpt-5" {
		t.Fatalf("models = %#v", models)
	}
}

func TestListProviderModelsUsesLocalCodexModelsForChatGPTOAuth(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	if err := saveAuthProfiles(AuthProfiles{Profiles: map[string]AuthProfile{
		"chatgpt": {
			Provider:    "openai-chatgpt",
			AuthType:    "oauth",
			AccessToken: "oauth-access-token",
			ExpiresAt:   time.Now().Add(time.Hour),
		},
	}}); err != nil {
		t.Fatal(err)
	}
	oldClient := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = oldClient })
	http.DefaultClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		t.Fatalf("openai-chatgpt model list should not make HTTP request to %s", r.URL.String())
		return nil, nil
	})}

	models, recommended, total, err := listProviderModels(context.Background(), Config{
		Provider:        "openai-chatgpt",
		AuthProfile:     "chatgpt",
		Model:           "gpt-5.4-mini",
		PreferredModels: defaultPreferredModels(),
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	if recommended != "gpt-5.5" || total != len(codexModels()) {
		t.Fatalf("recommended=%q total=%d models=%#v", recommended, total, models)
	}
	if !containsString(models, "gpt-5.4-mini") || !containsString(models, "gpt-5.3-codex") {
		t.Fatalf("models missing codex model: %#v", models)
	}
}

func TestOpenAIErrorMessageAcceptsStringAndObject(t *testing.T) {
	if got := openAIErrorMessage([]byte(`{"error":"bad key"}`)); got != "bad key" {
		t.Fatalf("string error = %q", got)
	}
	if got := openAIErrorMessage([]byte(`{"error":{"message":"bad scope"}}`)); got != "bad scope" {
		t.Fatalf("object error = %q", got)
	}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func TestCallModelUsesOpenAIResponsesForAPIKeyProvider(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	if err := saveAuthProfiles(AuthProfiles{Profiles: map[string]AuthProfile{
		"test": {Provider: "openai", AuthType: "api_key", APIKey: "sk-test"},
	}}); err != nil {
		t.Fatal(err)
	}
	oldClient := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = oldClient })
	http.DefaultClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.String() != "https://example.test/responses" {
			t.Fatalf("url = %q, want responses", r.URL.String())
		}
		if got := r.Header.Get("Authorization"); got != "Bearer sk-test" {
			t.Fatalf("Authorization = %q", got)
		}
		var raw map[string]any
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			t.Fatal(err)
		}
		if raw["model"] != "gpt-5.4" || raw["stream"] != true {
			t.Fatalf("unexpected responses request: %#v", raw)
		}
		if _, ok := raw["messages"]; ok {
			t.Fatalf("responses request should not use chat completions messages: %#v", raw)
		}
		reasoning, ok := raw["reasoning"].(map[string]any)
		if !ok || reasoning["effort"] != "high" {
			t.Fatalf("reasoning = %#v", raw["reasoning"])
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body: io.NopCloser(strings.NewReader(strings.Join([]string{
				`data: {"type":"response.output_item.done","item":{"type":"reasoning","summary":[{"type":"summary_text","text":"checked context"}]}}`,
				``,
				`data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}}`,
				``,
				`data: {"type":"response.completed","response":{}}`,
				``,
			}, "\n"))),
		}, nil
	})}

	msg, err := callModel(context.Background(), Config{
		Provider:       "openai",
		AuthProfile:    "test",
		OpenAIBaseURL:  "https://example.test",
		Model:          "gpt-5.4",
		ReasoningLevel: "high",
	}, []ChatMessage{
		{Role: "system", Content: "system"},
		{Role: "user", Content: "hello"},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content != "done" {
		t.Fatalf("content = %q", msg.Content)
	}
	if reasoning := reasoningTextFromMessage(msg); reasoning != "checked context" {
		t.Fatalf("reasoning = %q", reasoning)
	}
}

func TestCallOpenAIRequestBodyIncludesCompatibleReasoning(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	if err := saveAuthProfiles(AuthProfiles{Profiles: map[string]AuthProfile{
		"test": {Provider: "openai-compat", AuthType: "api_key", APIKey: "sk-test"},
	}}); err != nil {
		t.Fatal(err)
	}
	oldClient := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = oldClient })
	http.DefaultClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.String() != "https://example.test/chat/completions" {
			t.Fatalf("url = %q, want chat completions", r.URL.String())
		}
		if got := r.Header.Get("Authorization"); got != "Bearer sk-test" {
			t.Fatalf("Authorization = %q", got)
		}
		var req OpenAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req.Model != "gpt-5.4" {
			t.Fatalf("model = %q", req.Model)
		}
		if req.ReasoningEffort != "high" {
			t.Fatalf("reasoning_effort = %q", req.ReasoningEffort)
		}
		if len(req.Tools) != 1 || req.Tools[0].Function.Name != "read" {
			t.Fatalf("tools not sent correctly: %#v", req.Tools)
		}
		if len(req.Messages) != 2 || req.Messages[0].Role != "system" || req.Messages[1].Role != "user" {
			t.Fatalf("messages not sent correctly: %#v", req.Messages)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{"choices":[{"message":{"role":"assistant","reasoning_summary":"reviewed request","content":"done"}}]}`)),
		}, nil
	})}

	msg, err := callOpenAI(context.Background(), Config{
		Provider:       "openai-compat",
		AuthProfile:    "test",
		OpenAIBaseURL:  "https://example.test",
		Model:          "gpt-5.4",
		ReasoningLevel: "high",
	}, []ChatMessage{
		{Role: "system", Content: "system"},
		{Role: "user", Content: "hello"},
	}, []ToolSchema{{
		Type: "function",
		Function: ToolFunctionSpec{
			Name:        "read",
			Description: "read file",
			Parameters:  map[string]any{"type": "object"},
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content != "done" {
		t.Fatalf("content = %q", msg.Content)
	}
	if reasoning := reasoningTextFromMessage(msg); reasoning != "reviewed request" {
		t.Fatalf("reasoning = %q", reasoning)
	}
}

func TestCallOpenAIOmitsReasoningForNonReasoningModel(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	if err := saveAuthProfiles(AuthProfiles{Profiles: map[string]AuthProfile{
		"test": {Provider: "openai-compat", AuthType: "api_key", APIKey: "sk-test"},
	}}); err != nil {
		t.Fatal(err)
	}
	oldClient := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = oldClient })
	http.DefaultClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		var raw map[string]any
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			t.Fatal(err)
		}
		if _, ok := raw["reasoning_effort"]; ok {
			t.Fatalf("reasoning_effort should be omitted: %#v", raw)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{"choices":[{"message":{"role":"assistant","content":"done"}}]}`)),
		}, nil
	})}

	_, err := callOpenAI(context.Background(), Config{
		Provider:       "openai-compat",
		AuthProfile:    "test",
		OpenAIBaseURL:  "https://example.test",
		Model:          "gpt-4.1-mini",
		ReasoningLevel: "high",
	}, []ChatMessage{{Role: "user", Content: "hello"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
}

func TestCallOllamaIncludesThinkingForReasoningLevel(t *testing.T) {
	oldClient := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = oldClient })
	http.DefaultClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.String() != "https://ollama.test/api/chat" {
			t.Fatalf("url = %q, want ollama chat", r.URL.String())
		}
		var raw map[string]any
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			t.Fatal(err)
		}
		if raw["think"] != true {
			t.Fatalf("think = %#v, want true in %#v", raw["think"], raw)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{"message":{"role":"assistant","thinking":"checked files","content":"done"},"done":true}`)),
		}, nil
	})}

	msg, err := callOllama(context.Background(), Config{
		Provider:       "ollama",
		OllamaURL:      "https://ollama.test",
		Model:          "qwen3:latest",
		ReasoningLevel: "medium",
	}, []ChatMessage{{Role: "user", Content: "hello"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content != "done" {
		t.Fatalf("content = %q", msg.Content)
	}
	if reasoning := reasoningTextFromMessage(msg); reasoning != "checked files" {
		t.Fatalf("reasoning = %q", reasoning)
	}
}

func TestCallOllamaOmitsThinkingForNoneOrBlankReasoningLevel(t *testing.T) {
	tests := []struct {
		name  string
		level string
	}{
		{name: "none", level: "none"},
		{name: "blank", level: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			oldClient := http.DefaultClient
			t.Cleanup(func() { http.DefaultClient = oldClient })
			http.DefaultClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				var raw map[string]any
				if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
					t.Fatal(err)
				}
				if _, ok := raw["think"]; ok {
					t.Fatalf("think should be omitted: %#v", raw)
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader(`{"message":{"role":"assistant","content":"done"},"done":true}`)),
				}, nil
			})}

			_, err := callOllama(context.Background(), Config{
				Provider:       "ollama",
				OllamaURL:      "https://ollama.test",
				Model:          "qwen3:latest",
				ReasoningLevel: tt.level,
			}, []ChatMessage{{Role: "user", Content: "hello"}}, nil)
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCallOpenAIRejectsOAuthBearerWithoutGeneratedAPIKey(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	if err := saveAuthProfiles(AuthProfiles{Profiles: map[string]AuthProfile{
		"chatgpt": {
			Provider:    "openai-chatgpt",
			AuthType:    "oauth",
			AccessToken: "oauth-access-token",
			AccountID:   "account-123",
			ExpiresAt:   time.Now().Add(time.Hour),
		},
	}}); err != nil {
		t.Fatal(err)
	}

	_, err := callOpenAI(context.Background(), Config{
		Provider:       "openai-chatgpt",
		AuthProfile:    "chatgpt",
		OpenAIBaseURL:  "https://example.test",
		Model:          "gpt-4.1-mini",
		ReasoningLevel: "medium",
	}, []ChatMessage{{Role: "user", Content: "hello"}}, nil)
	if err == nil {
		t.Fatal("expected oauth bearer rejection")
	}
	if !strings.Contains(err.Error(), "Responses backend") {
		t.Fatalf("error = %q", err.Error())
	}
}

func TestCallModelUsesChatGPTCodexResponsesForOAuth(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	if err := saveAuthProfiles(AuthProfiles{Profiles: map[string]AuthProfile{
		"chatgpt": {
			Provider:    "openai-chatgpt",
			AuthType:    "oauth",
			AccessToken: "oauth-access-token",
			AccountID:   "account-123",
			ExpiresAt:   time.Now().Add(time.Hour),
		},
	}}); err != nil {
		t.Fatal(err)
	}
	oldClient := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = oldClient })
	http.DefaultClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.String() != "https://chatgpt.test/backend-api/codex/responses" {
			t.Fatalf("url = %s", r.URL.String())
		}
		if got := r.Header.Get("Authorization"); got != "Bearer oauth-access-token" {
			t.Fatalf("Authorization = %q", got)
		}
		if got := r.Header.Get("ChatGPT-Account-Id"); got != "account-123" {
			t.Fatalf("ChatGPT-Account-Id = %q", got)
		}
		var raw map[string]any
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			t.Fatal(err)
		}
		if raw["instructions"] != "system" {
			t.Fatalf("instructions = %#v", raw["instructions"])
		}
		if raw["tool_choice"] != "auto" || raw["stream"] != true {
			t.Fatalf("unexpected responses request: %#v", raw)
		}
		tools, ok := raw["tools"].([]any)
		if !ok || len(tools) != 1 {
			t.Fatalf("tools = %#v", raw["tools"])
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body: io.NopCloser(strings.NewReader(strings.Join([]string{
				`data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I'll write it."}]}}`,
				``,
				`data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"write","arguments":"{\"path\":\"main.go\",\"content\":\"package main\"}"}}`,
				``,
				`data: {"type":"response.completed","response":{}}`,
				``,
			}, "\n"))),
		}, nil
	})}

	msg, err := callModel(context.Background(), Config{
		Provider:             "openai-chatgpt",
		AuthProfile:          "chatgpt",
		ChatGPTCodexBaseURL:  "https://chatgpt.test/backend-api/codex",
		Model:                "gpt-5",
		ReasoningLevel:       "medium",
		ChatGPTOAuthClientID: defaultChatGPTOAuthClient,
	}, []ChatMessage{
		{Role: "system", Content: "system"},
		{Role: "user", Content: "hello"},
	}, []ToolSchema{{
		Type: "function",
		Function: ToolFunctionSpec{
			Name:        "write",
			Description: "write file",
			Parameters:  map[string]any{"type": "object"},
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content != "I'll write it." {
		t.Fatalf("content = %q", msg.Content)
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].ID != "call_1" || msg.ToolCalls[0].Function.Name != "write" {
		t.Fatalf("tool calls = %#v", msg.ToolCalls)
	}
}

func TestParseResponsesStreamTextDeltas(t *testing.T) {
	out, err := parseResponsesStream([]byte(strings.Join([]string{
		`data: {"type":"response.output_text.delta","delta":"hello"}`,
		``,
		`data: {"type":"response.output_text.delta","delta":" world"}`,
		``,
		`data: {"type":"response.completed","response":{}}`,
		``,
	}, "\n")))
	if err != nil {
		t.Fatal(err)
	}
	msg := chatMessageFromResponses(out)
	if msg.Content != "hello world" {
		t.Fatalf("content = %q", msg.Content)
	}
}

func TestParseResponsesStreamReasoningSummary(t *testing.T) {
	out, err := parseResponsesStream([]byte(strings.Join([]string{
		`data: {"type":"response.output_item.done","item":{"type":"reasoning","summary":[{"type":"summary_text","text":"inspected context"}]}}`,
		``,
		`data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}}`,
		``,
	}, "\n")))
	if err != nil {
		t.Fatal(err)
	}
	msg := chatMessageFromResponses(out)
	if msg.Content != "done" {
		t.Fatalf("content = %q", msg.Content)
	}
	if reasoning := reasoningTextFromMessage(msg); reasoning != "inspected context" {
		t.Fatalf("reasoning = %q", reasoning)
	}
}

func TestAssistantMsgPublishesReasoningBeforeContentWhenConfigured(t *testing.T) {
	sessionPath := filepath.Join(t.TempDir(), "session.json")
	m := model{
		cfg:         Config{SystemPrompt: "system", RecentHistoryMessages: 10, IncludeReasoningInContext: true},
		sessionPath: sessionPath,
		messages:    []ChatMessage{{Role: "system", Content: "system"}, {Role: "user", Content: "question"}},
		busy:        true,
	}
	updated, cmd := m.Update(assistantMsg{
		msg: ChatMessage{
			Role:     "assistant",
			Thinking: json.RawMessage(`"thought through it"`),
			Content:  "answer",
		},
	})
	if cmd != nil {
		t.Fatal("assistant message should not start async work")
	}
	got := updated.(model)
	if len(got.messages) != 4 {
		t.Fatalf("messages = %#v", got.messages)
	}
	if got.messages[2].Role != "assistant" || got.messages[2].Content != "thought through it" {
		t.Fatalf("reasoning message = %#v", got.messages[2])
	}
	if got.messages[3].Role != "assistant" || got.messages[3].Content != "answer" || len(got.messages[3].Thinking) != 0 {
		t.Fatalf("content message = %#v", got.messages[3])
	}
	if len(got.logs) < 2 || got.logs[0].Text != "thought through it" || got.logs[1].Text != "answer" {
		t.Fatalf("logs = %#v", got.logs)
	}
	context := got.contextMessages()
	if len(context) < 4 || context[len(context)-2].Content != "thought through it" || context[len(context)-1].Content != "answer" {
		t.Fatalf("context = %#v", context)
	}
}

func TestResponsesInputIncludesEmptyToolOutput(t *testing.T) {
	messages := []ChatMessage{
		{Role: "system", Content: "system"},
		{Role: "assistant", ToolCalls: []ToolCall{{
			ID: "call_1",
			Function: ToolFunction{
				Name:      "bash",
				Arguments: json.RawMessage(`{"command":"true"}`),
			},
		}}},
		{Role: "tool", ToolName: "bash", ToolCallID: "call_1", Content: ""},
	}
	body, err := json.Marshal(responsesRequest{Model: "gpt-5", Input: responsesInput(messages)})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), `"type":"function_call_output"`) {
		t.Fatalf("missing function call output item: %s", body)
	}
	if !strings.Contains(string(body), `"output":""`) {
		t.Fatalf("empty tool output was omitted: %s", body)
	}
	if strings.Contains(string(body), `"type":"message","role":"user","output"`) {
		t.Fatalf("message input included output field: %s", body)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
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

func TestSlashCommandsPersistReasoningFastAndModelAuto(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	m := model{
		cfg: Config{
			Model:                 "gpt-4.1-mini",
			Provider:              "openai",
			AuthProfile:           defaultAuthProfile,
			PreferredModels:       defaultPreferredModels(),
			ReasoningLevel:        "medium",
			RecentHistoryMessages: 10,
			OpenAIBaseURL:         "https://api.openai.com/v1",
			OpenAIAPIKeyEnv:       "OPENAI_API_KEY",
			ChatGPTIssuer:         defaultChatGPTIssuer,
			ChatGPTOAuthClientID:  defaultChatGPTOAuthClient,
		},
		configPath:  configPath,
		sessionPath: filepath.Join(dir, ".codegollm", "session.json"),
	}

	updated, cmd := m.handleSlashCommand("/reasoning high")
	if cmd != nil {
		t.Fatal("reasoning command should not start async work")
	}
	got := updated.(model)
	if got.cfg.ReasoningLevel != "high" {
		t.Fatalf("reasoning = %q", got.cfg.ReasoningLevel)
	}
	cfg, err := loadConfigFromDir(t, dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ReasoningLevel != "high" {
		t.Fatalf("saved reasoning = %q", cfg.ReasoningLevel)
	}

	updated, cmd = got.handleSlashCommand("/fast on")
	if cmd != nil {
		t.Fatal("fast command should not start async work")
	}
	got = updated.(model)
	if !got.cfg.Fast {
		t.Fatal("fast should be enabled")
	}
	cfg, err = loadConfigFromDir(t, dir)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.Fast {
		t.Fatal("saved fast should be enabled")
	}

	updated, cmd = got.handleSlashCommand("/model auto")
	if cmd != nil {
		t.Fatal("model auto command should not start async work")
	}
	got = updated.(model)
	if got.cfg.Model != "auto" {
		t.Fatalf("model = %q", got.cfg.Model)
	}
	cfg, err = loadConfigFromDir(t, dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Model != "auto" {
		t.Fatalf("saved model = %q", cfg.Model)
	}
}

func TestSlashCommandsRejectInvalidReasoningAndFastValue(t *testing.T) {
	dir := t.TempDir()
	m := model{
		cfg: Config{
			Model:                 "gpt-4.1-mini",
			Provider:              "openai",
			ReasoningLevel:        "medium",
			RecentHistoryMessages: 10,
		},
		configPath:  filepath.Join(dir, "config.yaml"),
		sessionPath: filepath.Join(dir, ".codegollm", "session.json"),
	}
	updated, cmd := m.handleSlashCommand("/reasoning maximum")
	if cmd != nil {
		t.Fatal("invalid reasoning command should not start async work")
	}
	got := updated.(model)
	if got.cfg.ReasoningLevel != "medium" {
		t.Fatalf("reasoning changed unexpectedly: %q", got.cfg.ReasoningLevel)
	}
	if len(got.logs) == 0 || got.logs[len(got.logs)-1].Kind != "error" {
		t.Fatalf("expected error log: %#v", got.logs)
	}

	updated, cmd = got.handleSlashCommand("/fast maybe")
	if cmd != nil {
		t.Fatal("invalid fast command should not start async work")
	}
	got = updated.(model)
	if got.cfg.Fast {
		t.Fatal("fast changed unexpectedly")
	}
	if len(got.logs) == 0 || !strings.Contains(got.logs[len(got.logs)-1].Text, "usage: /fast") {
		t.Fatalf("expected fast usage log: %#v", got.logs)
	}
}

func loadConfigFromDir(t *testing.T, dir string) (Config, error) {
	t.Helper()
	oldWd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.Chdir(oldWd) }()
	cfg, _, err := loadConfig()
	return cfg, err
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

func TestContextMessagesDoesNotStartWithOrphanToolResult(t *testing.T) {
	m := model{
		cfg: Config{SystemPrompt: "system", RecentHistoryMessages: 4},
		messages: []ChatMessage{
			{Role: "system", Content: "system"},
			{Role: "user", Content: "old"},
			{Role: "assistant", ToolCalls: []ToolCall{
				{ID: "call_1", Function: ToolFunction{Name: "write", Arguments: json.RawMessage(`{"path":"a","content":"a"}`)}},
				{ID: "call_2", Function: ToolFunction{Name: "write", Arguments: json.RawMessage(`{"path":"b","content":"b"}`)}},
			}},
			{Role: "tool", ToolName: "write", ToolCallID: "call_1", Content: "wrote a"},
			{Role: "tool", ToolName: "write", ToolCallID: "call_2", Content: "wrote b"},
			{Role: "user", Content: "next"},
		},
	}
	context := m.contextMessages()
	if context[1].Role != "assistant" || len(context[1].ToolCalls) != 2 {
		t.Fatalf("context should include assistant tool_calls before tool results: %#v", context)
	}
	if context[2].Role != "tool" || context[3].Role != "tool" {
		t.Fatalf("context lost tool results: %#v", context)
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

func TestOpenAIMessagesConvertsOrphanToolWithID(t *testing.T) {
	got := openAIMessages([]ChatMessage{
		{Role: "tool", ToolName: "write", ToolCallID: "call_1", Content: "wrote file"},
	})
	if got[0].Role != "user" || !strings.Contains(got[0].Content, "Prior tool result") {
		t.Fatalf("orphan tool should be converted to user context: %#v", got[0])
	}
}

func TestOpenAIMessagesRepairsDeniedToolCallShape(t *testing.T) {
	got := openAIMessages([]ChatMessage{
		{Role: "assistant", ToolCalls: []ToolCall{
			{ID: "call_1", Function: ToolFunction{Name: "bash", Arguments: json.RawMessage(`{"command":"deno task start"}`)}},
		}},
		{Role: "user", Content: "Do not run that command"},
	})
	if len(got) != 3 {
		t.Fatalf("len(got) = %d, want 3: %#v", len(got), got)
	}
	if got[0].Role != "assistant" || len(got[0].ToolCalls) != 1 {
		t.Fatalf("missing assistant tool call: %#v", got)
	}
	if got[1].Role != "tool" || got[1].ToolCallID != "call_1" {
		t.Fatalf("missing synthetic tool response before user message: %#v", got)
	}
	if got[2].Role != "user" {
		t.Fatalf("user message order changed: %#v", got)
	}
}

func TestNextUnansweredToolCallFindsSecondCall(t *testing.T) {
	first := ToolCall{ID: "call_1", Function: ToolFunction{Name: "write", Arguments: json.RawMessage(`{"path":"a","content":"a"}`)}}
	second := ToolCall{ID: "call_2", Function: ToolFunction{Name: "write", Arguments: json.RawMessage(`{"path":"b","content":"b"}`)}}
	m := model{messages: []ChatMessage{
		{Role: "system", Content: "system"},
		{Role: "assistant", ToolCalls: []ToolCall{first, second}},
		{Role: "tool", ToolName: "write", ToolCallID: "call_1", Content: "wrote a"},
	}}
	call, ok := m.nextUnansweredToolCall()
	if !ok {
		t.Fatal("expected unanswered tool call")
	}
	if call.ID != "call_2" {
		t.Fatalf("next call id = %q, want call_2", call.ID)
	}
}

func TestRepairUnansweredToolCallsAddsSyntheticToolMessage(t *testing.T) {
	messages := []ChatMessage{
		{Role: "system", Content: "system"},
		{Role: "assistant", ToolCalls: []ToolCall{
			{ID: "call_1", Function: ToolFunction{Name: "write"}},
			{ID: "call_2", Function: ToolFunction{Name: "write"}},
		}},
		{Role: "tool", ToolName: "write", ToolCallID: "call_1", Content: "wrote first"},
		{Role: "user", Content: "carry on"},
	}
	repaired, count := repairUnansweredToolCalls(messages)
	if count != 1 {
		t.Fatalf("repaired %d calls, want 1", count)
	}
	if repaired[2].Role != "tool" || repaired[2].ToolCallID != "call_1" {
		t.Fatalf("existing tool message order changed: %#v", repaired)
	}
	if repaired[3].Role != "tool" || repaired[3].ToolCallID != "call_2" {
		t.Fatalf("missing synthetic tool message: %#v", repaired)
	}
	if repaired[4].Role != "user" {
		t.Fatalf("user message order changed: %#v", repaired)
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

func TestToolResultsAreTruncatedForStorageAndContext(t *testing.T) {
	large := strings.Repeat("a", maxToolResultBytes+2048)
	truncated := truncateToolResult(large)
	if len(truncated) >= len(large) {
		t.Fatal("tool result was not truncated")
	}
	if !strings.Contains(truncated, "tool output truncated") {
		t.Fatalf("missing truncation marker: %q", truncated)
	}

	req := toolRequest{
		Call: ToolCall{ID: "call_1", Function: ToolFunction{Name: "bash"}},
		Name: "bash",
	}
	m := model{
		runID:       1,
		busy:        true,
		sessionPath: filepath.Join(t.TempDir(), ".codegollm", "session.json"),
		cfg:         Config{SystemPrompt: "system", RecentHistoryMessages: 10},
		messages: []ChatMessage{
			{Role: "system", Content: "system"},
			{Role: "assistant", ToolCalls: []ToolCall{req.Call}},
		},
	}
	updated, _ := m.Update(toolResultMsg{runID: 1, req: req, result: large})
	got := updated.(model)
	if len(got.messages) < 3 || len(got.messages[2].Content) >= len(large) {
		t.Fatalf("stored tool result was not truncated")
	}

	context := got.contextMessages()
	for _, msg := range context {
		if msg.Role == "tool" && len(msg.Content) >= len(large) {
			t.Fatal("context contains oversized tool result")
		}
	}
}

func TestRenderLogLineWrapsLongAssistantText(t *testing.T) {
	m := model{width: 48}
	rendered := m.renderLogLine(logLine{
		Kind: "assistant",
		Text: "Enhanced the Boid draw method in script.js to make each boid colored by its velocity direction.",
	})
	if !strings.Contains(rendered, "\n") {
		t.Fatalf("expected wrapped output, got %q", rendered)
	}
	if !strings.Contains(rendered, "velocity direction") {
		t.Fatalf("wrapped output lost text: %q", rendered)
	}
	lines := strings.Split(rendered, "\n")
	if !strings.HasPrefix(lines[0], "assistant: ") {
		t.Fatalf("first line missing label: %q", lines[0])
	}
	if strings.HasPrefix(lines[1], "assistant: ") {
		t.Fatalf("continuation line should not repeat label: %q", lines[1])
	}
}

func TestPromptHistoryNavigationAndClear(t *testing.T) {
	input := textinput.New()
	m := model{input: input}
	m = m.addPromptHistory("first prompt")
	m = m.addPromptHistory("second prompt")
	m.input.SetValue("draft")

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyUp})
	if cmd != nil {
		t.Fatal("history navigation should not return a command")
	}
	got := updated.(model)
	if got.input.Value() != "second prompt" {
		t.Fatalf("up value = %q", got.input.Value())
	}

	updated, _ = got.Update(tea.KeyMsg{Type: tea.KeyUp})
	got = updated.(model)
	if got.input.Value() != "first prompt" {
		t.Fatalf("second up value = %q", got.input.Value())
	}

	updated, _ = got.Update(tea.KeyMsg{Type: tea.KeyDown})
	got = updated.(model)
	if got.input.Value() != "second prompt" {
		t.Fatalf("down value = %q", got.input.Value())
	}

	updated, _ = got.Update(tea.KeyMsg{Type: tea.KeyDown})
	got = updated.(model)
	if got.input.Value() != "draft" {
		t.Fatalf("down to draft value = %q", got.input.Value())
	}

	updated, _ = got.Update(tea.KeyMsg{Type: tea.KeyCtrlU})
	got = updated.(model)
	if got.input.Value() != "" || got.promptHistoryPos != -1 {
		t.Fatalf("clear left value=%q pos=%d", got.input.Value(), got.promptHistoryPos)
	}
}

func TestSubmittedSlashCommandIsAddedToPromptHistory(t *testing.T) {
	input := textinput.New()
	input.SetValue("/help")
	m := model{input: input, promptHistoryPos: -1}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd != nil {
		t.Fatal("help command should not return async command")
	}
	got := updated.(model)
	if len(got.promptHistory) != 1 || got.promptHistory[0] != "/help" {
		t.Fatalf("history = %#v", got.promptHistory)
	}
	if got.input.Value() != "" {
		t.Fatalf("input value = %q", got.input.Value())
	}
}

func TestEscInterruptsBusyRun(t *testing.T) {
	m := model{sessionPath: filepath.Join(t.TempDir(), ".codegollm", "session.json")}
	m, runID, _ := m.beginRun(nil)

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if cmd != nil {
		t.Fatal("interrupt should not return a command")
	}
	got := updated.(model)
	if got.busy {
		t.Fatal("model should no longer be busy")
	}
	if got.runID == runID {
		t.Fatal("run id should advance so stale messages are ignored")
	}
	if len(got.logs) == 0 || !strings.Contains(got.logs[len(got.logs)-1].Text, "interrupted") {
		t.Fatalf("missing interrupt log: %#v", got.logs)
	}
}

func TestWorkingTickAnimatesOnlyCurrentBusyRun(t *testing.T) {
	m := model{}
	m, runID, _ := m.beginRun(nil)
	if m.workingText() != "working   " {
		t.Fatalf("initial working text = %q", m.workingText())
	}

	updated, cmd := m.Update(workingTickMsg{runID: runID})
	if cmd == nil {
		t.Fatal("current busy tick should schedule another tick")
	}
	got := updated.(model)
	if got.workingFrame != 1 || got.workingText() != "working.  " {
		t.Fatalf("working frame=%d text=%q", got.workingFrame, got.workingText())
	}

	updated, cmd = got.Update(workingTickMsg{runID: runID + 1})
	if cmd != nil {
		t.Fatal("stale tick should not schedule another tick")
	}
	got = updated.(model)
	if got.workingFrame != 1 {
		t.Fatalf("stale tick changed frame to %d", got.workingFrame)
	}
}

func TestEscInterruptsRunningToolWithToolResponse(t *testing.T) {
	req := toolRequest{
		Call: ToolCall{ID: "call_1", Function: ToolFunction{Name: "bash"}},
		Name: "bash",
		Args: map[string]string{"command": "sleep 60"},
	}
	m := model{
		sessionPath: filepath.Join(t.TempDir(), ".codegollm", "session.json"),
		messages: []ChatMessage{
			{Role: "system", Content: "system"},
			{Role: "assistant", ToolCalls: []ToolCall{req.Call}},
		},
	}
	m, _, _ = m.beginRun(&req)

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	got := updated.(model)
	if got.busy {
		t.Fatal("model should no longer be busy")
	}
	if len(got.messages) != 3 {
		t.Fatalf("len(messages) = %d, want 3: %#v", len(got.messages), got.messages)
	}
	tool := got.messages[2]
	if tool.Role != "tool" || tool.ToolCallID != "call_1" || !strings.Contains(tool.Content, "interrupted") {
		t.Fatalf("missing interrupted tool response: %#v", tool)
	}
}

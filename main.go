package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"gopkg.in/yaml.v3"
)

const (
	configFile  = "config.yaml"
	sessionFile = ".codegollm/session.json"
)

type Config struct {
	Model                 string   `yaml:"model"`
	Provider              string   `yaml:"provider"`
	OllamaURL             string   `yaml:"ollama_url"`
	OpenAIBaseURL         string   `yaml:"openai_base_url"`
	OpenAIAPIKeyEnv       string   `yaml:"openai_api_key_env"`
	RecentHistoryMessages int      `yaml:"recent_history_messages"`
	ApprovedTools         []string `yaml:"approved_tools"`
	SystemPrompt          string   `yaml:"system_prompt"`
}

type OllamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

type ChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolName   string     `json:"tool_name,omitempty"`
}

type ToolCall struct {
	ID       string          `json:"id,omitempty"`
	Type     string          `json:"type,omitempty"`
	Function ToolFunction    `json:"function"`
	RawArgs  json.RawMessage `json:"-"`
}

type ToolFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type ToolSchema struct {
	Type     string           `json:"type"`
	Function ToolFunctionSpec `json:"function"`
}

type ToolFunctionSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type OllamaRequest struct {
	Model    string        `json:"model"`
	Stream   bool          `json:"stream"`
	Messages []ChatMessage `json:"messages"`
	Tools    []ToolSchema  `json:"tools"`
}

type OllamaResponse struct {
	Message ChatMessage `json:"message"`
	Done    bool        `json:"done"`
	Error   string      `json:"error,omitempty"`
}

type OpenAIChatRequest struct {
	Model    string              `json:"model"`
	Messages []OpenAIChatMessage `json:"messages"`
	Tools    []ToolSchema        `json:"tools,omitempty"`
}

type OpenAIChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type OpenAIChatResponse struct {
	Choices []struct {
		Message ChatMessage `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
}

type toolRequest struct {
	Call    ToolCall
	Name    string
	Args    map[string]string
	Preview string
	Diff    string
}

type logLine struct {
	Kind string
	Text string
}

type SessionState struct {
	Messages       []ChatMessage `json:"messages"`
	Summary        string        `json:"summary"`
	SummaryThrough int           `json:"summary_through"`
	UpdatedAt      time.Time     `json:"updated_at"`
}

type model struct {
	cfg            Config
	configPath     string
	sessionPath    string
	root           string
	input          textinput.Model
	messages       []ChatMessage
	summary        string
	summaryThrough int
	logs           []logLine
	busy           bool
	pending        *toolRequest
	steering       *toolRequest
	modelChoices   []string
	modelCursor    int
	err            error
}

type assistantMsg struct{ msg ChatMessage }
type toolResultMsg struct{ result string }
type summaryMsg struct {
	summary string
	through int
	err     error
}
type modelsMsg struct{ models []string }
type errMsg struct{ err error }

var (
	titleStyle     = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("212"))
	userStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("86"))
	assistantStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("229"))
	toolStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("111"))
	errStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("203"))
	addStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("42"))
	delStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("203"))
	metaStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("244"))
	boxStyle       = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).Padding(0, 1)
)

func main() {
	cfg, configPath, err := loadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(1)
	}
	root, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "cwd: %v\n", err)
		os.Exit(1)
	}
	root, _ = filepath.Abs(root)

	envLogs := loadEnvFiles(root, configPath)

	ti := textinput.New()
	ti.Placeholder = "Ask codegollm to change code..."
	ti.Focus()
	ti.CharLimit = 4096
	ti.Width = 80

	sessionPath := filepath.Join(root, sessionFile)
	messages := []ChatMessage{{
		Role:    "system",
		Content: cfg.SystemPrompt,
	}}
	var summary string
	var summaryThrough int
	logs := []logLine{{Kind: "system", Text: fmt.Sprintf("workspace: %s | provider: %s | model: %s", root, cfg.Provider, cfg.Model)}}
	logs = append(logs, envLogs...)
	if session, err := loadSession(sessionPath); err == nil {
		if len(session.Messages) > 0 {
			messages = ensureSystemMessage(session.Messages, cfg.SystemPrompt)
		}
		summary = session.Summary
		summaryThrough = session.SummaryThrough
		logs = append(logs, logLine{Kind: "system", Text: "resumed session from " + sessionPath})
	} else if !errors.Is(err, os.ErrNotExist) {
		logs = append(logs, logLine{Kind: "error", Text: "could not load session: " + err.Error()})
	}

	m := model{
		cfg:            cfg,
		configPath:     configPath,
		sessionPath:    sessionPath,
		root:           root,
		input:          ti,
		messages:       messages,
		summary:        summary,
		summaryThrough: summaryThrough,
		logs:           logs,
	}
	m = m.saveSession()

	if _, err := tea.NewProgram(m).Run(); err != nil {
		fmt.Fprintf(os.Stderr, "tui: %v\n", err)
		os.Exit(1)
	}
}

func loadConfig() (Config, string, error) {
	cfg := Config{
		Model:                 "gpt-4.1-mini",
		Provider:              "openai",
		OllamaURL:             "http://localhost:11434",
		OpenAIBaseURL:         "https://api.openai.com/v1",
		OpenAIAPIKeyEnv:       "OPENAI_API_KEY",
		RecentHistoryMessages: 10,
		SystemPrompt:          "You are codegollm, a minimal local coding agent with read, write, edit, and bash tools.",
	}
	path := findConfigPath()
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return cfg, path, nil
	}
	if err != nil {
		return cfg, path, err
	}
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return cfg, path, err
	}
	providerSet := yamlHasKey(data, "provider")
	if cfg.Model == "" {
		cfg.Model = "gpt-4.1-mini"
	}
	if cfg.Provider == "" {
		cfg.Provider = "openai"
	}
	if !providerSet && looksLikeOllamaModel(cfg.Model) {
		cfg.Provider = "ollama"
	}
	if cfg.OllamaURL == "" {
		cfg.OllamaURL = "http://localhost:11434"
	}
	if cfg.OpenAIBaseURL == "" {
		cfg.OpenAIBaseURL = "https://api.openai.com/v1"
	}
	if cfg.OpenAIAPIKeyEnv == "" {
		cfg.OpenAIAPIKeyEnv = "OPENAI_API_KEY"
	}
	if cfg.RecentHistoryMessages <= 0 {
		cfg.RecentHistoryMessages = 10
	}
	return cfg, path, nil
}

func yamlHasKey(data []byte, key string) bool {
	var raw map[string]any
	if err := yaml.Unmarshal(data, &raw); err != nil {
		return false
	}
	_, ok := raw[key]
	return ok
}

func looksLikeOllamaModel(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	if strings.Contains(model, "/") || strings.Contains(model, ":") {
		return true
	}
	prefixes := []string{"gemma", "llama", "mistral", "qwen", "deepseek", "phi", "codellama"}
	for _, prefix := range prefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

func findConfigPath() string {
	cwdConfig := filepath.Join(".", configFile)
	if _, err := os.Stat(cwdConfig); err == nil {
		return cwdConfig
	}
	exe, err := os.Executable()
	if err == nil {
		exeConfig := filepath.Join(filepath.Dir(exe), configFile)
		if _, err := os.Stat(exeConfig); err == nil {
			return exeConfig
		}
	}
	return cwdConfig
}

func saveConfig(path string, cfg Config) error {
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func loadEnvFiles(root, configPath string) []logLine {
	var logs []logLine
	seen := map[string]bool{}
	paths := []string{
		filepath.Join(root, ".env"),
		filepath.Join(filepath.Dir(configPath), ".env"),
	}
	for _, path := range paths {
		abs, err := filepath.Abs(path)
		if err != nil {
			continue
		}
		if seen[abs] {
			continue
		}
		seen[abs] = true
		loaded, err := loadEnvFile(abs)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			logs = append(logs, logLine{Kind: "error", Text: "could not load " + abs + ": " + err.Error()})
			continue
		}
		if loaded > 0 {
			logs = append(logs, logLine{Kind: "system", Text: fmt.Sprintf("loaded %d environment variable(s) from %s", loaded, abs)})
		}
	}
	return logs
}

func loadEnvFile(path string) (int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	loaded := 0
	for lineNo, line := range strings.Split(string(data), "\n") {
		key, value, ok, err := parseEnvLine(line)
		if err != nil {
			return loaded, fmt.Errorf("line %d: %w", lineNo+1, err)
		}
		if !ok {
			continue
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		if err := os.Setenv(key, value); err != nil {
			return loaded, err
		}
		loaded++
	}
	return loaded, nil
}

func parseEnvLine(line string) (string, string, bool, error) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return "", "", false, nil
	}
	line = strings.TrimPrefix(line, "export ")
	parts := strings.SplitN(line, "=", 2)
	if len(parts) != 2 {
		return "", "", false, errors.New("expected KEY=value")
	}
	key := strings.TrimSpace(parts[0])
	if key == "" {
		return "", "", false, errors.New("empty key")
	}
	value := strings.TrimSpace(parts[1])
	quoted := false
	if len(value) >= 2 {
		quote := value[0]
		if (quote == '"' || quote == '\'') && value[len(value)-1] == quote {
			value = value[1 : len(value)-1]
			quoted = true
		}
	}
	if !quoted {
		if hash := strings.Index(value, " #"); hash >= 0 {
			value = strings.TrimSpace(value[:hash])
		}
	}
	return key, value, true, nil
}

func loadSession(path string) (SessionState, error) {
	var state SessionState
	data, err := os.ReadFile(path)
	if err != nil {
		return state, err
	}
	if err := json.Unmarshal(data, &state); err != nil {
		return state, err
	}
	return state, nil
}

func saveSession(path string, state SessionState) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	state.UpdatedAt = time.Now()
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func ensureSystemMessage(messages []ChatMessage, systemPrompt string) []ChatMessage {
	copied := append([]ChatMessage(nil), messages...)
	if len(copied) == 0 {
		return []ChatMessage{{Role: "system", Content: systemPrompt}}
	}
	if copied[0].Role == "system" {
		copied[0].Content = systemPrompt
		return copied
	}
	return append([]ChatMessage{{Role: "system", Content: systemPrompt}}, copied...)
}

func (m model) saveSession() model {
	if err := saveSession(m.sessionPath, SessionState{
		Messages:       m.messages,
		Summary:        m.summary,
		SummaryThrough: m.summaryThrough,
	}); err != nil {
		m.logs = append(m.logs, logLine{Kind: "error", Text: "could not save session: " + err.Error()})
	}
	return m
}

func (m model) Init() tea.Cmd {
	return textinput.Blink
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		if m.steering != nil {
			switch msg.String() {
			case "enter":
				instruction := strings.TrimSpace(m.input.Value())
				if instruction == "" {
					return m, nil
				}
				req := *m.steering
				m.steering = nil
				m.input.SetValue("")
				m.input.Placeholder = "Ask codegollm to change code..."
				m.busy = true
				m.logs = append(m.logs, logLine{Kind: "user", Text: "instead of " + formatTool(req) + ": " + instruction})
				m.messages = append(m.messages, ChatMessage{
					Role: "user",
					Content: "Do not run the proposed tool call " + formatTool(req) + ". Instead: " + instruction +
						". Continue using tools as needed.",
				})
				m = m.saveSession()
				return m, callModelCmd(m.cfg, m.contextMessages())
			case "esc":
				m.pending = m.steering
				m.steering = nil
				m.input.SetValue("")
				m.input.Placeholder = "Ask codegollm to change code..."
				return m, nil
			case "ctrl+c":
				return m, tea.Quit
			}
			var cmd tea.Cmd
			m.input, cmd = m.input.Update(msg)
			return m, cmd
		}
		if len(m.modelChoices) > 0 {
			switch msg.String() {
			case "up", "k":
				if m.modelCursor > 0 {
					m.modelCursor--
				}
			case "down", "j":
				if m.modelCursor < len(m.modelChoices)-1 {
					m.modelCursor++
				}
			case "enter":
				selected := m.modelChoices[m.modelCursor]
				m.cfg.Model = selected
				m.modelChoices = nil
				m.modelCursor = 0
				if err := saveConfig(m.configPath, m.cfg); err != nil {
					m.logs = append(m.logs, logLine{Kind: "error", Text: "model switched to " + selected + " but config save failed: " + err.Error()})
				} else {
					m.logs = append(m.logs, logLine{Kind: "system", Text: "model switched to " + selected + " and saved to " + m.configPath})
				}
			case "esc", "ctrl+c":
				m.modelChoices = nil
				m.modelCursor = 0
			}
			return m, nil
		}
		if m.pending != nil {
			switch msg.String() {
			case "y", "enter":
				req := *m.pending
				m.pending = nil
				m.busy = true
				m.logs = append(m.logs, logLine{Kind: "tool", Text: "approved " + formatTool(req)})
				return m, runToolCmd(m.root, req)
			case "a":
				req := *m.pending
				m.pending = nil
				m.busy = true
				key := approvalKey(req)
				if !hasApproval(m.cfg.ApprovedTools, key) {
					m.cfg.ApprovedTools = append(m.cfg.ApprovedTools, key)
					if err := saveConfig(m.configPath, m.cfg); err != nil {
						m.logs = append(m.logs, logLine{Kind: "error", Text: "approval saved in memory but config save failed: " + err.Error()})
					}
				}
				m.logs = append(m.logs, logLine{Kind: "tool", Text: "approved always " + formatTool(req)})
				return m, runToolCmd(m.root, req)
			case "n", "esc":
				req := *m.pending
				m.pending = nil
				m.steering = &req
				m.input.SetValue("")
				m.input.Placeholder = "Tell codegollm what to do differently..."
				m.input.Focus()
				m.logs = append(m.logs, logLine{Kind: "system", Text: "denied approval; enter revised instruction"})
				return m, nil
			case "ctrl+c":
				return m, tea.Quit
			}
			return m, nil
		}
		switch msg.String() {
		case "ctrl+c":
			return m, tea.Quit
		case "enter":
			if m.busy {
				return m, nil
			}
			prompt := strings.TrimSpace(m.input.Value())
			if prompt == "" {
				return m, nil
			}
			m.input.SetValue("")
			if strings.HasPrefix(prompt, "/") {
				return m.handleSlashCommand(prompt)
			}
			m.busy = true
			m.messages = append(m.messages, ChatMessage{Role: "user", Content: prompt})
			m.logs = append(m.logs, logLine{Kind: "user", Text: prompt})
			m = m.saveSession()
			return m, callModelCmd(m.cfg, m.contextMessages())
		}
	case assistantMsg:
		m.busy = false
		m.messages = append(m.messages, msg.msg)
		if msg.msg.Content != "" {
			m.logs = append(m.logs, logLine{Kind: "assistant", Text: msg.msg.Content})
		}
		m = m.saveSession()
		if len(msg.msg.ToolCalls) > 0 {
			req, err := m.prepareTool(msg.msg.ToolCalls[0])
			if err != nil {
				m.busy = true
				rejected := reqFromCall(msg.msg.ToolCalls[0])
				result := fmt.Sprintf("tool rejected: %s. Tool %q must use valid JSON object arguments. Raw arguments: %s", err.Error(), rejected.Name, string(msg.msg.ToolCalls[0].Function.Arguments))
				return m, continueWithToolResult(rejected, result)
			}
			if req.Name == "read" || hasApproval(m.cfg.ApprovedTools, approvalKey(req)) {
				m.busy = true
				m.logs = append(m.logs, logLine{Kind: "tool", Text: "running " + formatTool(req)})
				return m, runToolCmd(m.root, req)
			}
			m.pending = &req
			m.logs = append(m.logs, logLine{Kind: "tool", Text: "approval requested for " + formatTool(req)})
			return m, nil
		}
		if shouldForceToolUse(msg.msg.Content) {
			nudge := ChatMessage{
				Role: "user",
				Content: strings.Join([]string{
					"Do not ask me to provide code, file contents, patches, commands, or implementation details.",
					"You are the coding agent. Continue by using your tools to inspect the workspace, write the needed code, and run relevant checks.",
					"Use project lint, format, and test commands when available; add suitable lint tooling if the project lacks it and the language warrants it.",
					"Only ask a question if the requirement is genuinely ambiguous and cannot be resolved by reading files or making a reasonable implementation choice.",
				}, " "),
			}
			m.messages = append(m.messages, nudge)
			m.logs = append(m.logs, logLine{Kind: "system", Text: "prompted model to continue with tools"})
			m = m.saveSession()
			m.busy = true
			return m, callModelCmd(m.cfg, m.contextMessages())
		}
		return m, m.maybeSummarizeCmd()
	case toolResultMsg:
		m.busy = true
		toolName, toolCallID := m.lastTool()
		m.messages = append(m.messages, ChatMessage{Role: "tool", Content: msg.result, ToolName: toolName, ToolCallID: toolCallID})
		m.logs = append(m.logs, logLine{Kind: "tool", Text: previewContent(strings.TrimSpace(msg.result))})
		m = m.saveSession()
		return m, callModelCmd(m.cfg, m.contextMessages())
	case summaryMsg:
		if msg.err != nil {
			m.logs = append(m.logs, logLine{Kind: "error", Text: msg.err.Error()})
		} else if msg.through > m.summaryThrough {
			m.summary = msg.summary
			m.summaryThrough = msg.through
			m.logs = append(m.logs, logLine{Kind: "system", Text: "updated history summary"})
			m = m.saveSession()
		}
	case modelsMsg:
		m.busy = false
		if len(msg.models) == 0 {
			m.logs = append(m.logs, logLine{Kind: "error", Text: "no Ollama models found"})
			return m, nil
		}
		m.modelChoices = msg.models
		m.modelCursor = 0
		for i, name := range msg.models {
			if name == m.cfg.Model {
				m.modelCursor = i
				break
			}
		}
	case errMsg:
		m.busy = false
		m.err = msg.err
		m.logs = append(m.logs, logLine{Kind: "error", Text: msg.err.Error()})
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

func (m model) View() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("CodeGollm"))
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("root: %s | provider: %s | model: %s | saved approvals: %d\n\n", m.root, m.cfg.Provider, m.cfg.Model, len(m.cfg.ApprovedTools)))

	start := 0
	if len(m.logs) > 18 {
		start = len(m.logs) - 18
	}
	for _, l := range m.logs[start:] {
		style := assistantStyle
		label := "assistant"
		switch l.Kind {
		case "user":
			style = userStyle
			label = "you"
		case "tool":
			style = toolStyle
			label = "tool"
		case "error":
			style = errStyle
			label = "error"
		case "system":
			style = titleStyle
			label = "system"
		}
		b.WriteString(style.Render(label + ": "))
		b.WriteString(l.Text)
		b.WriteString("\n\n")
	}

	if m.steering != nil {
		b.WriteString(boxStyle.Render("Tell codegollm what to do differently for:\n\n" + formatTool(*m.steering) + "\n\n[enter] send instruction  [esc] back to approval"))
		b.WriteString("\n\n")
	} else if m.pending != nil {
		help := "[y/enter] approve once  [a] always approve exact operation  [n/esc] deny and tell me what to do differently"
		approval := "Approval required\n\n" + m.pending.Preview
		if m.pending.Diff != "" {
			approval += "\n\n" + renderDiff(m.pending.Diff)
		}
		approval += "\n\n" + help
		b.WriteString(boxStyle.Render(approval))
		b.WriteString("\n\n")
	} else if len(m.modelChoices) > 0 {
		var list strings.Builder
		list.WriteString("Select model\n\n")
		for i, name := range m.modelChoices {
			cursor := "  "
			if i == m.modelCursor {
				cursor = "> "
			}
			list.WriteString(cursor)
			list.WriteString(name)
			if name == m.cfg.Model {
				list.WriteString("  current")
			}
			list.WriteString("\n")
		}
		list.WriteString("\n[up/down] move  [enter] select  [esc] cancel")
		b.WriteString(boxStyle.Render(list.String()))
		b.WriteString("\n\n")
	} else if m.busy {
		b.WriteString(toolStyle.Render("working..."))
		b.WriteString("\n\n")
	}

	b.WriteString(m.input.View())
	b.WriteString("\n")
	b.WriteString("enter sends | ctrl+c quits")
	return b.String()
}

func (m model) contextMessages() []ChatMessage {
	recent := m.cfg.RecentHistoryMessages
	if recent <= 0 {
		recent = 10
	}
	context := []ChatMessage{{Role: "system", Content: m.cfg.SystemPrompt}}
	if strings.TrimSpace(m.summary) != "" {
		context = append(context, ChatMessage{
			Role:    "system",
			Content: "Conversation and workspace history summary:\n" + m.summary,
		})
	}
	start := len(m.messages) - recent
	if start < 1 {
		start = 1
	}
	if start < m.summaryThrough {
		start = m.summaryThrough
	}
	context = append(context, m.messages[start:]...)
	return context
}

func (m model) maybeSummarizeCmd() tea.Cmd {
	recent := m.cfg.RecentHistoryMessages
	if recent <= 0 {
		recent = 10
	}
	through := len(m.messages) - recent
	if through <= 1 || through <= m.summaryThrough {
		return nil
	}
	messages := append([]ChatMessage(nil), m.messages[1:through]...)
	existing := m.summary
	cfg := m.cfg
	return summarizeCmd(cfg, existing, messages, through)
}

func (m model) handleSlashCommand(prompt string) (tea.Model, tea.Cmd) {
	fields := strings.Fields(prompt)
	cmd := fields[0]
	switch cmd {
	case "/model":
		if strings.ToLower(strings.TrimSpace(m.cfg.Provider)) != "ollama" {
			m.logs = append(m.logs, logLine{Kind: "system", Text: "current provider is " + m.cfg.Provider + "; set model in config.yaml"})
			return m, nil
		}
		m.busy = true
		m.logs = append(m.logs, logLine{Kind: "system", Text: "loading Ollama models..."})
		return m, listModelsCmd(m.cfg)
	case "/help":
		m.logs = append(m.logs, logLine{Kind: "system", Text: "commands: /model, /help"})
		return m, nil
	default:
		m.logs = append(m.logs, logLine{Kind: "error", Text: "unknown command: " + cmd})
		return m, nil
	}
}

func (m model) prepareTool(call ToolCall) (toolRequest, error) {
	req := reqFromCall(call)
	args, err := parseToolArgs(call.Function.Arguments)
	if err != nil {
		return req, err
	}
	req.Args = args
	switch req.Name {
	case "read":
		path, err := cleanPath(m.root, argValue(req.Args, "path", "file_path", "filepath", "file", "filename"))
		if err != nil {
			return req, err
		}
		req.Args["path"] = path
		req.Preview = "read " + path
	case "write":
		path, err := cleanPath(m.root, argValue(req.Args, "path", "file_path", "filepath", "file", "filename"))
		if err != nil {
			return req, err
		}
		content := argValue(req.Args, "content", "contents", "text", "body", "data")
		if content == "" {
			return req, errors.New("content is empty")
		}
		req.Args["path"] = path
		req.Args["content"] = content
		req.Preview = "write " + path
		req.Diff = m.writePreviewDiff(path, content)
	case "edit":
		path, err := cleanPath(m.root, argValue(req.Args, "path", "file_path", "filepath", "file", "filename"))
		if err != nil {
			return req, err
		}
		oldText := argValue(req.Args, "old", "old_text", "before", "search", "find")
		newText := argValue(req.Args, "new", "new_text", "after", "replace", "replacement")
		if oldText == "" {
			return req, errors.New("old text is empty")
		}
		req.Args["path"] = path
		req.Args["old"] = oldText
		req.Args["new"] = newText
		req.Preview = "edit " + path
		req.Diff = replacementDiff(oldText, newText)
	case "bash":
		command := argValue(req.Args, "command", "cmd", "shell", "script")
		if strings.TrimSpace(command) == "" {
			return req, errors.New("bash command is empty")
		}
		req.Args["command"] = command
		req.Preview = "$ " + req.Args["command"]
	default:
		return req, errors.New("unknown tool: " + req.Name)
	}
	return req, nil
}

func reqFromCall(call ToolCall) toolRequest {
	return toolRequest{Call: call, Name: call.Function.Name, Args: map[string]string{}}
}

func parseToolArgs(raw json.RawMessage) (map[string]string, error) {
	args := map[string]string{}
	if len(raw) == 0 || string(raw) == "null" {
		return args, nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w; raw=%s", err, string(raw))
	}
	if s, ok := value.(string); ok {
		if err := json.Unmarshal([]byte(s), &value); err != nil {
			return nil, fmt.Errorf("arguments were a string, not a JSON object: %s", s)
		}
	}
	flattenArgs("", value, args)
	return args, nil
}

func flattenArgs(prefix string, value any, out map[string]string) {
	switch v := value.(type) {
	case map[string]any:
		for key, nested := range v {
			normalized := normalizeArgName(key)
			if prefix == "" || prefix == "input" || prefix == "args" || prefix == "arguments" {
				flattenArgs(normalized, nested, out)
			} else {
				flattenArgs(prefix+"_"+normalized, nested, out)
			}
		}
	case string:
		if prefix != "" {
			out[prefix] = v
		}
	case float64, bool:
		if prefix != "" {
			out[prefix] = fmt.Sprint(v)
		}
	}
}

func normalizeArgName(name string) string {
	name = strings.TrimSpace(strings.ToLower(name))
	name = strings.ReplaceAll(name, "-", "_")
	name = strings.ReplaceAll(name, " ", "_")
	return name
}

func argValue(args map[string]string, names ...string) string {
	for _, name := range names {
		if value := strings.TrimSpace(args[normalizeArgName(name)]); value != "" {
			return value
		}
	}
	return ""
}

func formatTool(req toolRequest) string {
	switch req.Name {
	case "read", "write", "edit":
		return req.Name + " " + req.Args["path"]
	case "bash":
		return "bash " + req.Args["command"]
	default:
		return req.Name
	}
}

func approvalKey(req toolRequest) string {
	switch req.Name {
	case "read":
		return "read:" + req.Args["path"]
	case "write":
		return "write:" + req.Args["path"] + ":" + req.Args["content"]
	case "edit":
		return "edit:" + req.Args["path"] + ":" + req.Args["old"] + "=>" + req.Args["new"]
	case "bash":
		return "bash:" + strings.TrimSpace(req.Args["command"])
	default:
		return req.Name + ":" + fmt.Sprint(req.Args)
	}
}

func hasApproval(approvals []string, key string) bool {
	for _, approval := range approvals {
		if approval == key {
			return true
		}
	}
	return false
}

func cleanPath(root, p string) (string, error) {
	if strings.TrimSpace(p) == "" {
		return "", errors.New("path is empty")
	}
	if filepath.IsAbs(p) {
		return "", errors.New("absolute paths are not allowed")
	}
	abs, err := filepath.Abs(filepath.Join(root, p))
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(root, abs)
	if err != nil {
		return "", err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", errors.New("path escapes workspace root")
	}
	return rel, nil
}

func previewContent(s string) string {
	if len(s) > 3000 {
		return s[:3000] + "\n... truncated ..."
	}
	return s
}

func (m model) writePreviewDiff(path, content string) string {
	current, err := os.ReadFile(filepath.Join(m.root, path))
	if err != nil {
		return addedFileDiff(content)
	}
	return simpleLineDiff(string(current), content)
}

func addedFileDiff(content string) string {
	var b strings.Builder
	b.WriteString("--- /dev/null\n")
	b.WriteString("+++ proposed\n")
	for _, line := range splitLines(content) {
		b.WriteString("+")
		b.WriteString(line)
		b.WriteString("\n")
	}
	return previewContent(b.String())
}

func replacementDiff(oldText, newText string) string {
	var b strings.Builder
	b.WriteString("--- old\n")
	b.WriteString("+++ new\n")
	for _, line := range splitLines(oldText) {
		b.WriteString("-")
		b.WriteString(line)
		b.WriteString("\n")
	}
	for _, line := range splitLines(newText) {
		b.WriteString("+")
		b.WriteString(line)
		b.WriteString("\n")
	}
	return previewContent(b.String())
}

func simpleLineDiff(oldText, newText string) string {
	oldLines := splitLines(oldText)
	newLines := splitLines(newText)
	if strings.Join(oldLines, "\n") == strings.Join(newLines, "\n") {
		return "no textual change"
	}
	table := make([][]int, len(oldLines)+1)
	for i := range table {
		table[i] = make([]int, len(newLines)+1)
	}
	for i := len(oldLines) - 1; i >= 0; i-- {
		for j := len(newLines) - 1; j >= 0; j-- {
			if oldLines[i] == newLines[j] {
				table[i][j] = table[i+1][j+1] + 1
			} else if table[i+1][j] >= table[i][j+1] {
				table[i][j] = table[i+1][j]
			} else {
				table[i][j] = table[i][j+1]
			}
		}
	}
	var b strings.Builder
	b.WriteString("--- current\n")
	b.WriteString("+++ proposed\n")
	i, j := 0, 0
	for i < len(oldLines) && j < len(newLines) {
		switch {
		case oldLines[i] == newLines[j]:
			b.WriteString(" ")
			b.WriteString(oldLines[i])
			b.WriteString("\n")
			i++
			j++
		case table[i+1][j] >= table[i][j+1]:
			b.WriteString("-")
			b.WriteString(oldLines[i])
			b.WriteString("\n")
			i++
		default:
			b.WriteString("+")
			b.WriteString(newLines[j])
			b.WriteString("\n")
			j++
		}
	}
	for ; i < len(oldLines); i++ {
		b.WriteString("-")
		b.WriteString(oldLines[i])
		b.WriteString("\n")
	}
	for ; j < len(newLines); j++ {
		b.WriteString("+")
		b.WriteString(newLines[j])
		b.WriteString("\n")
	}
	return previewContent(b.String())
}

func splitLines(s string) []string {
	s = strings.TrimSuffix(s, "\n")
	if s == "" {
		return []string{""}
	}
	return strings.Split(s, "\n")
}

func renderDiff(diff string) string {
	var b strings.Builder
	for _, line := range strings.Split(diff, "\n") {
		switch {
		case strings.HasPrefix(line, "+++") || strings.HasPrefix(line, "---"):
			b.WriteString(metaStyle.Render(line))
		case strings.HasPrefix(line, "+"):
			b.WriteString(addStyle.Render(line))
		case strings.HasPrefix(line, "-"):
			b.WriteString(delStyle.Render(line))
		default:
			b.WriteString(line)
		}
		b.WriteString("\n")
	}
	return strings.TrimSuffix(b.String(), "\n")
}

func shouldForceToolUse(content string) bool {
	text := strings.ToLower(content)
	if strings.TrimSpace(text) == "" {
		return false
	}
	phrases := []string{
		"please provide the code",
		"please provide code",
		"provide the code",
		"provide the file",
		"provide the contents",
		"provide the html",
		"provide the css",
		"provide the javascript",
		"send me the code",
		"paste the code",
		"you can replace",
		"you should replace",
		"you need to replace",
		"you can run",
		"you should run",
		"you need to run",
		"run the following",
		"copy this into",
		"save this as",
	}
	for _, phrase := range phrases {
		if strings.Contains(text, phrase) {
			return true
		}
	}
	return strings.Contains(text, "please") &&
		(strings.Contains(text, "provide") || strings.Contains(text, "paste")) &&
		(strings.Contains(text, "code") || strings.Contains(text, "file") || strings.Contains(text, "html") || strings.Contains(text, "javascript") || strings.Contains(text, "css"))
}

func (m model) lastTool() (string, string) {
	if len(m.messages) > 0 {
		last := m.messages[len(m.messages)-1]
		if len(last.ToolCalls) > 0 {
			return last.ToolCalls[0].Function.Name, last.ToolCalls[0].ID
		}
	}
	return "", ""
}

func continueWithToolResult(req toolRequest, result string) tea.Cmd {
	return func() tea.Msg {
		_ = req
		return toolResultMsg{result: result}
	}
}

func callModelCmd(cfg Config, messages []ChatMessage) tea.Cmd {
	return func() tea.Msg {
		msg, err := callModel(cfg, messages, toolSchemas())
		if err != nil {
			return errMsg{err}
		}
		return assistantMsg{msg: msg}
	}
}

func summarizeCmd(cfg Config, existing string, messages []ChatMessage, through int) tea.Cmd {
	return func() tea.Msg {
		var b strings.Builder
		if strings.TrimSpace(existing) != "" {
			b.WriteString("Existing summary:\n")
			b.WriteString(existing)
			b.WriteString("\n\n")
		}
		b.WriteString("Older messages to fold into the summary:\n")
		for _, msg := range messages {
			b.WriteString(formatMessageForSummary(msg))
			b.WriteString("\n")
		}
		reqMessages := []ChatMessage{
			{
				Role: "system",
				Content: strings.Join([]string{
					"Summarize a coding-agent session for future context.",
					"Keep it compact but preserve durable facts:",
					"user goals and preferences",
					"project/workspace state",
					"files changed or discussed",
					"commands run and important outcomes",
					"open tasks, blockers, and current direction",
					"Do not use Markdown tables or headings.",
				}, "\n"),
			},
			{Role: "user", Content: b.String()},
		}
		msg, err := callModel(cfg, reqMessages, nil)
		if err != nil {
			return summaryMsg{through: through, err: fmt.Errorf("summary update failed: %w", err)}
		}
		return summaryMsg{summary: strings.TrimSpace(msg.Content), through: through}
	}
}

func formatMessageForSummary(msg ChatMessage) string {
	switch msg.Role {
	case "assistant":
		if len(msg.ToolCalls) > 0 {
			var calls []string
			for _, call := range msg.ToolCalls {
				calls = append(calls, call.Function.Name+"("+string(call.Function.Arguments)+")")
			}
			return "assistant tool call: " + strings.Join(calls, "; ")
		}
		return "assistant: " + previewContent(msg.Content)
	case "tool":
		return "tool " + msg.ToolName + ": " + previewContent(msg.Content)
	default:
		return msg.Role + ": " + previewContent(msg.Content)
	}
}

func callModel(cfg Config, messages []ChatMessage, tools []ToolSchema) (ChatMessage, error) {
	switch strings.ToLower(strings.TrimSpace(cfg.Provider)) {
	case "", "openai":
		return callOpenAI(cfg, messages, tools)
	case "ollama":
		return callOllama(cfg, messages, tools)
	default:
		return ChatMessage{}, errors.New("unknown provider: " + cfg.Provider)
	}
}

func callOpenAI(cfg Config, messages []ChatMessage, tools []ToolSchema) (ChatMessage, error) {
	apiKey := strings.TrimSpace(os.Getenv(cfg.OpenAIAPIKeyEnv))
	if apiKey == "" {
		return ChatMessage{}, fmt.Errorf("missing OpenAI API key: set %s in your shell environment", cfg.OpenAIAPIKeyEnv)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	body, err := json.Marshal(OpenAIChatRequest{
		Model:    cfg.Model,
		Messages: openAIMessages(messages),
		Tools:    tools,
	})
	if err != nil {
		return ChatMessage{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(cfg.OpenAIBaseURL, "/")+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return ChatMessage{}, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return ChatMessage{}, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return ChatMessage{}, err
	}
	var out OpenAIChatResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return ChatMessage{}, err
	}
	if resp.StatusCode >= 300 {
		if out.Error != nil && out.Error.Message != "" {
			return ChatMessage{}, fmt.Errorf("openai status %d: %s", resp.StatusCode, out.Error.Message)
		}
		return ChatMessage{}, fmt.Errorf("openai status %d: %s", resp.StatusCode, string(data))
	}
	if out.Error != nil && out.Error.Message != "" {
		return ChatMessage{}, errors.New(out.Error.Message)
	}
	if len(out.Choices) == 0 {
		return ChatMessage{}, errors.New("openai returned no choices")
	}
	msg := out.Choices[0].Message
	for i := range msg.ToolCalls {
		if msg.ToolCalls[i].Type == "" {
			msg.ToolCalls[i].Type = "function"
		}
	}
	return msg, nil
}

func openAIMessages(messages []ChatMessage) []OpenAIChatMessage {
	out := make([]OpenAIChatMessage, 0, len(messages))
	for _, msg := range messages {
		if msg.Role == "tool" && msg.ToolCallID == "" {
			out = append(out, OpenAIChatMessage{
				Role:    "user",
				Content: "Prior tool result from " + msg.ToolName + ": " + msg.Content,
			})
			continue
		}
		if msg.Role == "assistant" && hasToolCallWithoutID(msg.ToolCalls) {
			out = append(out, OpenAIChatMessage{
				Role:    "assistant",
				Content: formatMessageForSummary(msg),
			})
			continue
		}
		converted := OpenAIChatMessage{
			Role:       msg.Role,
			Content:    msg.Content,
			ToolCalls:  openAIToolCalls(msg.ToolCalls),
			ToolCallID: msg.ToolCallID,
		}
		out = append(out, converted)
	}
	return out
}

func hasToolCallWithoutID(calls []ToolCall) bool {
	for _, call := range calls {
		if strings.TrimSpace(call.ID) == "" {
			return true
		}
	}
	return false
}

func openAIToolCalls(calls []ToolCall) []ToolCall {
	if len(calls) == 0 {
		return nil
	}
	converted := make([]ToolCall, len(calls))
	for i, call := range calls {
		converted[i] = call
		if converted[i].Type == "" {
			converted[i].Type = "function"
		}
		args := strings.TrimSpace(string(converted[i].Function.Arguments))
		if args != "" && !strings.HasPrefix(args, "\"") {
			encoded, err := json.Marshal(args)
			if err == nil {
				converted[i].Function.Arguments = encoded
			}
		}
	}
	return converted
}

func callOllama(cfg Config, messages []ChatMessage, tools []ToolSchema) (ChatMessage, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	body, err := json.Marshal(OllamaRequest{
		Model:    cfg.Model,
		Stream:   false,
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		return ChatMessage{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(cfg.OllamaURL, "/")+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return ChatMessage{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return ChatMessage{}, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return ChatMessage{}, err
	}
	if resp.StatusCode >= 300 {
		return ChatMessage{}, fmt.Errorf("ollama status %d: %s", resp.StatusCode, string(data))
	}
	var out OllamaResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return ChatMessage{}, err
	}
	if out.Error != "" {
		return ChatMessage{}, errors.New(out.Error)
	}
	return out.Message, nil
}

func listModelsCmd(cfg Config) tea.Cmd {
	return func() tea.Msg {
		models, err := listModelsFromCLI()
		if err != nil {
			models, err = listModelsFromAPI(cfg)
		}
		if err != nil {
			return errMsg{err: err}
		}
		return modelsMsg{models: models}
	}
}

func listModelsFromCLI() ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "ollama", "list")
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	var models []string
	for i, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || i == 0 {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) > 0 {
			models = append(models, fields[0])
		}
	}
	if len(models) == 0 {
		return nil, errors.New("ollama list returned no models")
	}
	return models, nil
}

func listModelsFromAPI(cfg Config) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(cfg.OllamaURL, "/")+"/api/tags", nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("ollama tags status %d: %s", resp.StatusCode, string(data))
	}
	var tags OllamaTagsResponse
	if err := json.Unmarshal(data, &tags); err != nil {
		return nil, err
	}
	var models []string
	for _, model := range tags.Models {
		if model.Name != "" {
			models = append(models, model.Name)
		}
	}
	if len(models) == 0 {
		return nil, errors.New("Ollama returned no models")
	}
	return models, nil
}

func runToolCmd(root string, req toolRequest) tea.Cmd {
	return func() tea.Msg {
		result, err := runTool(root, req)
		if err != nil {
			result = "error: " + err.Error()
		}
		return toolResultMsg{result: result}
	}
}

func runTool(root string, req toolRequest) (string, error) {
	switch req.Name {
	case "read":
		data, err := os.ReadFile(filepath.Join(root, req.Args["path"]))
		if err != nil {
			return "", err
		}
		return string(data), nil
	case "write":
		path := filepath.Join(root, req.Args["path"])
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			return "", err
		}
		if err := os.WriteFile(path, []byte(req.Args["content"]), 0644); err != nil {
			return "", err
		}
		return "wrote " + req.Args["path"], nil
	case "edit":
		path := filepath.Join(root, req.Args["path"])
		data, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		oldText := req.Args["old"]
		newText := req.Args["new"]
		current := string(data)
		if oldText == "" {
			return "", errors.New("old text is empty")
		}
		if !strings.Contains(current, oldText) {
			return "", errors.New("old text not found")
		}
		updated := strings.Replace(current, oldText, newText, 1)
		if err := os.WriteFile(path, []byte(updated), 0644); err != nil {
			return "", err
		}
		return "edited " + req.Args["path"], nil
	case "bash":
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		cmd := exec.CommandContext(ctx, "bash", "-lc", req.Args["command"])
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if ctx.Err() == context.DeadlineExceeded {
			return string(out), errors.New("command timed out")
		}
		if err != nil {
			return string(out), fmt.Errorf("%w\n%s", err, string(out))
		}
		return string(out), nil
	default:
		return "", errors.New("unknown tool: " + req.Name)
	}
}

func toolSchemas() []ToolSchema {
	stringProp := func(desc string) map[string]any {
		return map[string]any{"type": "string", "description": desc}
	}
	object := func(props map[string]any, required []string) map[string]any {
		return map[string]any{
			"type":                 "object",
			"properties":           props,
			"required":             required,
			"additionalProperties": false,
		}
	}
	return []ToolSchema{
		{Type: "function", Function: ToolFunctionSpec{
			Name:        "read",
			Description: "Read a UTF-8 text file under the workspace root.",
			Parameters:  object(map[string]any{"path": stringProp("Relative file path to read.")}, []string{"path"}),
		}},
		{Type: "function", Function: ToolFunctionSpec{
			Name:        "write",
			Description: "Create or replace a UTF-8 text file under the workspace root.",
			Parameters: object(map[string]any{
				"path":    stringProp("Relative file path to write."),
				"content": stringProp("Complete file content."),
			}, []string{"path", "content"}),
		}},
		{Type: "function", Function: ToolFunctionSpec{
			Name:        "edit",
			Description: "Replace the first exact occurrence of old text with new text in a file under the workspace root.",
			Parameters: object(map[string]any{
				"path": stringProp("Relative file path to edit."),
				"old":  stringProp("Exact text to replace."),
				"new":  stringProp("Replacement text."),
			}, []string{"path", "old", "new"}),
		}},
		{Type: "function", Function: ToolFunctionSpec{
			Name:        "bash",
			Description: "Run a shell command from the workspace root.",
			Parameters:  object(map[string]any{"command": stringProp("Command to run with bash -lc.")}, []string{"command"}),
		}},
	}
}

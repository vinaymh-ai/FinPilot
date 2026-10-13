export type Severity = "info" | "warning" | "critical" | "positive";

export interface Category {
  id: string;
  label: string;
  group: string;
  color: string;
  icon: string;
  description: string;
}

export interface AgentCard {
  id: string;
  name: string;
  role: string;
  model: string;
}

export interface Meta {
  version: string;
  currency: string;
  user: { id: number; name: string };
  categories: Category[];
  agents: AgentCard[];
  llm_enabled: boolean;
  decisions?: string;
}

export interface Txn {
  id: number;
  date: string;
  time: string | null;
  description: string;
  merchant: string;
  amount: number;
  category: string | null;
  category_label: string;
  category_source: string | null;
  category_confidence: number | null;
  category_reason: string | null;
  tags: string[];
  is_recurring: boolean;
  is_anomaly: boolean;
  anomaly_reason: string | null;
  notes: string | null;
}

export interface Insight {
  id: number;
  run_id: number;
  kind: string;
  severity: Severity;
  title: string;
  detail: string;
  metrics: Record<string, unknown>;
  transaction_ids: number[];
}

export interface Recommendation {
  id: number;
  run_id: number;
  kind: string;
  title: string;
  message: string;
  action: string;
  monthly_impact: number;
  priority: number;
  urgency: "low" | "normal" | "high";
  status: "new" | "accepted" | "dismissed" | "done";
  trigger: string;
  evidence: Record<string, unknown>;
  reasoning: string[];
  generated_by: string;
  grounded: boolean;
  fingerprint: string;
  created_at: string | null;
}

export interface GoalProjection {
  id: number | null;
  name: string;
  target: number;
  saved: number;
  remaining: number;
  deadline: string;
  probability: number;
  status: "on_track" | "at_risk" | "off_track" | "achieved";
  projected_at_deadline: number;
  shortfall: number;
  eta: string | null;
  required_monthly: number;
  months_left: number;
  spend_at_deadline?: boolean;
}

export interface Goal {
  id: number;
  name: string;
  target_amount: number;
  saved_amount: number;
  deadline: string;
  priority: number;
  status: string;
  progress_pct: number;
  history: { date: string; run_id: number; probability: number; shortfall: number; eta: string | null }[];
  projection: GoalProjection | null;
  goal_type: string | null;
  spend_at_deadline: boolean | null;
  type_source: "jev" | "rules" | "user" | null;
  type_confidence: number | null;
}

export interface Checkpoint {
  days: number;
  date: string;
  p10: number;
  p50: number;
  p90: number;
}

export interface UpcomingEvent {
  date: string;
  merchant: string;
  amount: number;
  kind: string;
  category: string;
}

export interface ForecastSummary {
  as_of: string;
  start_balance: number;
  horizon_days: number;
  safety_buffer: number;
  low_point: { date: string; p50: number; p10: number };
  end_balance: { date: string; p10: number; p50: number; p90: number };
  prob_overdraft: number;
  prob_below_buffer: number;
  first_risk_date: string | null;
  checkpoints: Checkpoint[];
  goals: GoalProjection[];
  monthly: {
    month: string;
    income: number;
    commitments: number;
    variable_spend: number;
    net: number;
    partial: boolean;
    variable_by_category: Record<string, number>;
  }[];
  upcoming: UpcomingEvent[];
  summary: { net_change_90d_p50: number; avg_monthly_net: number; daily_variable_net: number };
  series?: { dates: string[]; p10: number[]; p50: number[]; p90: number[]; expected: number[] };
  category_monthly?: Record<string, number>;
  narrative?: string;
  narrative_model?: string;
}

export interface RunStep {
  id: number;
  agent: string;
  status: string;
  started_at: string;
  duration_ms: number;
  summary: string | null;
  details: Record<string, unknown>;
  llm_calls: number;
  error: string | null;
}

export interface AgentRun {
  id: number;
  trigger: string;
  graph: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  duration_ms: number | null;
  summary: Record<string, any>;
  error: string | null;
  steps?: RunStep[];
}

export interface Digest {
  id: number;
  run_id: number | null;
  trigger: string;
  subject: string;
  headline: string;
  action: string;
  body_markdown: string;
  deliveries: Record<string, { status: string; detail: string }>;
  created_at: string | null;
}

export interface Dashboard {
  overview: {
    as_of: string;
    currency: string;
    balance: number;
    last_30_days: { income: number; spend: number; savings_rate_pct: number | null };
    top_categories_30d: { category: string; amount: number }[];
    subscriptions: { count: number; monthly_cost: number };
    transactions: number;
    goals: { name: string; target: number; saved: number; deadline: string }[];
  };
  cashflow: { month: string; income: number; spend: number; invested: number; net: number }[];
  categories_30d: { category: string; label: string; amount: number }[];
  forecast: ForecastSummary | null;
  insights: Insight[];
  recommendations: Recommendation[];
  headline: string | null;
  profile: string[];
  last_run: AgentRun | null;
  analysis_running?: boolean;
  latest_digest: Digest | null;
  goals: Goal[];
}

export interface RecurringItem {
  id: number;
  merchant: string;
  category: string;
  category_label: string;
  kind: string;
  cadence: string;
  amount: number;
  signed_amount: number;
  monthly_cost: number;
  last_amount: number;
  first_date: string;
  last_date: string;
  next_expected: string;
  occurrences: number;
  regularity: number;
  price_change_pct: number | null;
  status: string;
  usage: string;
  usage_signal: string | null;
  unused?: boolean;
  overlap_with?: string[];
}

export interface Adjustment {
  type: "one_time" | "recurring" | "category_change" | "cancel_recurring" | "income_change";
  amount?: number;
  kind?: "expense" | "income";
  date?: string | null;
  end_date?: string | null;
  cadence?: string;
  category?: string | null;
  pct?: number | null;
  merchants?: string[];
  label?: string;
}

export interface SimulationResult {
  adjustments: (Adjustment & { description: string })[];
  verdict: "affordable" | "stretch" | "not_affordable";
  reasons: string[];
  baseline: ForecastSummary;
  scenario: ForecastSummary;
  delta: {
    end_balance_p50: number;
    low_point_p50: number;
    prob_overdraft: number;
    prob_below_buffer: number;
    goals: {
      name: string;
      prob_before: number;
      prob_after: number;
      eta_before: string | null;
      eta_after: string | null;
      delay_days: number | null;
      shortfall_before: number;
      shortfall_after: number;
    }[];
  };
  suggestions: { type: string; text: string; date?: string; weeks?: number }[];
}

export interface ChatTrace {
  tool: string;
  args: Record<string, unknown>;
  result: unknown;
}

export interface ChatRoute {
  intent: string;
  confidence: number;
  probabilities: Record<string, number>;
  slots: { category: string | null; period: string | null; purchase_kind: string | null };
  complexity: number;
  complexity_label: string;
  model: string;
  input_tokens: number;
  cost_usd: number;
  latency_ms: number;
  cached: boolean;
  order_checked: boolean;
  action: "answered_by_rules" | "escalated_to_llm" | "refused" | "fallback_rules" | "pending";
  llm_model: string | null;
}

export interface ChatMessage {
  id?: number;
  role: "user" | "assistant";
  content: string;
  trace?: ChatTrace[];
  model?: string;
  route?: ChatRoute | null;
  pending?: boolean;
}

export interface DecisionRow {
  id: number;
  run_id: number | null;
  agent: string;
  purpose: string;
  subject: string;
  question: string;
  answer: string;
  confidence: number | null;
  probabilities: Record<string, number>;
  action: "accepted" | "deferred" | "escalated" | "rejected" | "overridden";
  model: string;
  latency_ms: number;
  cached: boolean;
  created_at: string | null;
}

export interface KindUsage {
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  cache_hits: number;
  successes: number;
  avg_latency_ms: number;
}

export interface SavingsRow {
  purpose: string;
  jev_calls: number;
  jev_input_tokens: number;
  jev_cost_usd: number;
  decisions: number;
  actions: Record<string, number>;
  replaces_llm: boolean;
  llm_calls_avoided?: number;
  llm_equivalent_tokens?: { input: number; output: number };
  llm_cost_avoided_usd?: number;
}

export interface DecisionStats {
  by_kind: Record<string, KindUsage>;
  decisions: Record<string, { actions: Record<string, number>; total: number; avg_confidence: Record<string, number> }>;
  savings: {
    rows: SavingsRow[];
    llm_cost_avoided_usd: number;
    jev_cost_replacing_usd: number;
    net_savings_usd: number;
    jev_cost_added_checks_usd: number;
    assumptions: {
      llm_price_in_per_1m: number;
      llm_price_out_per_1m: number;
      jev_price_in_per_1m: number;
      llm_equivalent_tokens: Record<string, { input: number; output: number }>;
      note: string;
    };
  };
  decision_model: string;
}

export interface JevState {
  enabled: boolean;
  configured: boolean;
  available: boolean;
  provider: "mesh" | "typesafe";
  model: string;
  endpoint: string;
  key_env: string;
  overrides: Record<string, unknown>;
  accept_confidence: number;
  route_confidence: number;
  guard_confidence: number;
  escalate_to_llm: boolean;
  order_check: boolean;
  price_in_per_1m: number;
  used_for: { purpose: string; agent: string; replaces: string }[];
}

export interface ProviderInfo {
  name: string;
  label: string;
  configured: boolean;
  default_model: string;
  suggested_models: string[];
  free_tier: boolean;
  notes: string;
  base_url: string | null;
  key_env: string | null;
}

export interface LLMState {
  enabled: boolean;
  env_provider: string;
  overrides: {
    enabled?: boolean;
    provider?: string;
    model?: string;
    agents?: Record<string, { provider?: string; model?: string }>;
  };
  providers: ProviderInfo[];
  agents: Record<string, { resolved: string; fallbacks: string[] }>;
  fallback_providers: string[];
}

export interface SettingsView {
  profile: {
    name: string;
    currency: string;
    current_balance: number;
    balance_as_of: string;
    safety_buffer: number | null;
    digest_email: string | null;
    telegram_chat_id: string | null;
    digest_enabled: boolean;
  };
  llm: LLMState;
  decisions: JevState;
  notifications: { email_configured: boolean; telegram_configured: boolean };
}

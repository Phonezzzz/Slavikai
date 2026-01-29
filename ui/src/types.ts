export type Message = {
  role: "user" | "assistant" | "system";
  content: string;
};

export type DecisionOptionView = {
  id: string;
  title: string;
  action: string;
  risk: string;
};

export type DecisionPacketView = {
  id: string;
  reason: string;
  summary: string;
  options: DecisionOptionView[];
  default_option_id?: string | null;
};

export type UIEvent = {
  id: string;
  type: string;
  ts: string;
  payload: unknown;
};

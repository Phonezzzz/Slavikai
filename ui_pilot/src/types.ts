export type Message = {
  role: "user" | "assistant" | "system";
  content: string;
};

export type PilotEvent = {
  id: string;
  type: string;
  ts: string;
  payload: unknown;
};

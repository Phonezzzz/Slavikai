import { describe, expect, it, vi } from "vitest";

import { deliverTranscription } from "../components/canvas";

describe("Desktop voice routing", () => {
  it("sends STT text through the same Desktop request callback", async () => {
    const send = vi.fn().mockResolvedValue(true);
    const insert = vi.fn();

    const result = await deliverTranscription("  create a file  ", "desktop", send, insert);

    expect(result).toBe("sent");
    expect(send).toHaveBeenCalledWith({ content: "create a file", attachments: [] });
    expect(insert).not.toHaveBeenCalled();
  });

  it("keeps non-Desktop transcription in the composer", async () => {
    const send = vi.fn();
    const insert = vi.fn();

    const result = await deliverTranscription("chat text", "ask", send, insert);

    expect(result).toBe("composed");
    expect(send).not.toHaveBeenCalled();
    expect(insert).toHaveBeenCalledWith("chat text");
  });

  it("does not hide a rejected Desktop request", async () => {
    await expect(
      deliverTranscription("run", "desktop", vi.fn().mockResolvedValue(false), vi.fn()),
    ).rejects.toThrow("not accepted");
  });
});

import { describe, expect, it, vi } from "vitest";

import { deliverTranscription } from "../components/canvas";

describe("Voice transcription", () => {
  it("puts the transcription in the editable composer", () => {
    const insert = vi.fn();

    deliverTranscription("  create a file  ", insert);

    expect(insert).toHaveBeenCalledOnce();
    expect(insert).toHaveBeenCalledWith("  create a file  ");
  });

  it("rejects a blank transcription", () => {
    const insert = vi.fn();

    expect(() => deliverTranscription("  ", insert)).toThrow("STT returned empty text.");
    expect(insert).not.toHaveBeenCalled();
  });
});

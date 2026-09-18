import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import "./contact.css";

const interestOptions = [
  "Research collaboration",
  "Clinical partnership",
  "Technology / integration",
  "Investment",
  "Other",
] as const;

type Interest = (typeof interestOptions)[number];
type SubmissionStatus = "idle" | "submitting" | "success" | "error";

interface Enquiry {
  name: string;
  organisation: string;
  email: string;
  interests: Interest[];
}

const environment = import.meta.env as Record<string, string | boolean | undefined>;
const contactEmail = String(environment.VITE_CONTACT_EMAIL ?? "airest.operating@gmail.com").trim();
const contactEndpoint = String(environment.VITE_CONTACT_ENDPOINT ?? "").trim();

export function ContactForm() {
  const [enquiry, setEnquiry] = useState<Enquiry>({
    name: "",
    organisation: "",
    email: "",
    interests: [],
  });
  const [status, setStatus] = useState<SubmissionStatus>("idle");
  const [message, setMessage] = useState("");
  const activeRequest = useRef<AbortController | null>(null);
  const submitting = status === "submitting";

  useEffect(() => () => activeRequest.current?.abort(), []);

  function clearFeedback() {
    if (status !== "submitting") {
      setStatus("idle");
      setMessage("");
    }
  }

  function updateField(field: "name" | "organisation" | "email", value: string) {
    clearFeedback();
    setEnquiry((current) => ({ ...current, [field]: value }));
  }

  function toggleInterest(interest: Interest) {
    clearFeedback();
    setEnquiry((current) => ({
      ...current,
      interests: current.interests.includes(interest)
        ? current.interests.filter((selected) => selected !== interest)
        : [...current.interests, interest],
    }));
  }

  async function submitEnquiry(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting || activeRequest.current) return;

    const payload: Enquiry = {
      name: enquiry.name.trim(),
      organisation: enquiry.organisation.trim(),
      email: enquiry.email.trim(),
      interests: [...enquiry.interests],
    };

    if (contactEndpoint) {
      const controller = new AbortController();
      activeRequest.current = controller;
      const timeout = window.setTimeout(() => controller.abort(), 15000);
      setStatus("submitting");
      setMessage("Sending your enquiry…");

      try {
        const response = await fetch(contactEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        if (!response.ok) throw new Error("The enquiry could not be accepted.");

        setStatus("success");
        setMessage("Your enquiry has been sent. Thank you for reaching out.");
      } catch {
        setStatus("error");
        setMessage("We couldn’t send your enquiry. Your details are still here — please try again.");
      } finally {
        window.clearTimeout(timeout);
        activeRequest.current = null;
      }
      return;
    }

    if (contactEmail) {
      const subject = "An enquiry for AIREST";
      const body = [
        "Hello AIREST,",
        "",
        `Name: ${payload.name}`,
        `Organisation: ${payload.organisation || "Not provided"}`,
        `Email: ${payload.email}`,
        `Interested in: ${payload.interests.join(", ") || "A conversation"}`,
        "",
        "I’d like to connect with your team.",
      ].join("\n");

      window.location.href = `mailto:${encodeURIComponent(contactEmail)}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
      setStatus("success");
      setMessage("Your email draft is ready. Send it from your email app.");
      return;
    }

    setStatus("error");
    setMessage("We’re not accepting online enquiries just yet. Please try again later.");
  }

  return (
    <form
      className="contact-form-panel"
      onSubmit={submitEnquiry}
      aria-labelledby="contact-form-title"
      aria-busy={submitting}
    >
      <h2 className="contact-form-title" id="contact-form-title">
        Let’s connect<br />constellations
      </h2>

      <div className="contact-fields">
        <label className="contact-field">
          <span>Name <small>(required)</small></span>
          <input
            name="name"
            type="text"
            placeholder="Name"
            autoComplete="name"
            required
            pattern=".*\S.*"
            title="Please enter your name."
            maxLength={160}
            disabled={submitting}
            value={enquiry.name}
            onChange={(event) => updateField("name", event.target.value)}
          />
        </label>
        <label className="contact-field">
          <span>Organisation <small>(optional)</small></span>
          <input
            name="organisation"
            type="text"
            placeholder="Organisation"
            autoComplete="organization"
            maxLength={200}
            disabled={submitting}
            value={enquiry.organisation}
            onChange={(event) => updateField("organisation", event.target.value)}
          />
        </label>
        <label className="contact-field contact-field-email">
          <span>Email <small>(required)</small></span>
          <input
            name="email"
            type="email"
            placeholder="Email"
            autoComplete="email"
            required
            maxLength={254}
            disabled={submitting}
            value={enquiry.email}
            onChange={(event) => updateField("email", event.target.value)}
          />
        </label>
      </div>

      <fieldset className="contact-interests" disabled={submitting}>
        <legend>I’m interested in:</legend>
        <div className="contact-interest-options">
          {interestOptions.map((interest) => (
            <label className="contact-interest" key={interest}>
              <input
                className="contact-visually-hidden"
                type="checkbox"
                name="interests"
                value={interest}
                checked={enquiry.interests.includes(interest)}
                onChange={() => toggleInterest(interest)}
              />
              <span>{interest}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <div className="contact-submit-wrap">
        <button className="contact-submit" type="submit" disabled={submitting}>
          {submitting ? "Sending…" : "Let’s talk!"}
        </button>
        <p
          className="contact-form-status"
          data-status={status}
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          {message || (contactEndpoint ? "Your enquiry will be sent to AIREST." : "Opens a draft in your email app.")}
        </p>
      </div>
    </form>
  );
}

export default ContactForm;

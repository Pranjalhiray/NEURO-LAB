import { useState, type FormEvent } from "react";
import { ArrowRight, Brain, CalendarDays, LockKeyhole, Mail, Phone, ShieldCheck, UserRound } from "lucide-react";
import { IS_HOSTED_DEMO, type PatientDetails } from "./api";

type Props = {
  onPatientLogin: (patient: PatientDetails) => void | Promise<void>;
  onAdminLogin: (username: string, password: string) => Promise<void>;
};

type Portal = "patient" | "admin";

function ageOnDate(dateOfBirth: string) {
  const birth = new Date(`${dateOfBirth}T00:00:00`);
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  if (today.getMonth() < birth.getMonth() || (today.getMonth() === birth.getMonth() && today.getDate() < birth.getDate())) age -= 1;
  return age;
}

function localToday() {
  const today = new Date();
  return `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
}

export function PatientAccess({ onPatientLogin, onAdminLogin }: Props) {
  const [portal, setPortal] = useState<Portal>("patient");
  const [name, setName] = useState("");
  const [dateOfBirth, setDateOfBirth] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submitPatient = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const cleanName = name.trim();
    const today = localToday();
    if (!cleanName) { setError("Enter the patient's name to continue."); return; }
    if (!dateOfBirth || dateOfBirth > today) { setError("Enter a valid date of birth."); return; }
    const age = ageOnDate(dateOfBirth);
    if (age < 0 || age > 130) { setError("Enter a valid date of birth."); return; }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) { setError("Enter a valid email address."); return; }
    const cleanPhone = phone.trim();
    if (!/^[+\d][\d\s().-]{6,24}$/.test(cleanPhone) || cleanPhone.replace(/\D/g, "").length < 7) { setError("Enter a valid contact number."); return; }
    setError("");
    setBusy(true);
    try { await onPatientLogin({ name: cleanName, age, date_of_birth: dateOfBirth, email: email.trim(), phone: cleanPhone }); }
    catch (e) { setError(e instanceof Error ? e.message : "Could not open the patient session."); }
    finally { setBusy(false); }
  };

  const submitAdmin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    setBusy(true);
    try { await onAdminLogin(username.trim(), password); }
    catch (e) { setError(e instanceof Error ? e.message : "Admin sign-in failed."); }
    finally { setBusy(false); }
  };

  const changePortal = (next: Portal) => { setPortal(next); setError(""); };

  return <main className="patient-access-page">
    <header className="patient-brand"><span className="brand-mark"><Brain size={21} /></span><div><b>NEURO<span>/</span>LAB</b><small>LOCAL MRI WORKSTATION</small></div></header>
    <section className="patient-access-card">
      <div className="access-symbol">{portal === "patient" ? <UserRound size={22} /> : <ShieldCheck size={22} />}</div>
      <div className="eyebrow">SESSION DETAILS</div>
      <h1>{portal === "patient" ? "Patient sign-in" : "Administrator sign-in"}</h1>
      <div className="portal-tabs" role="tablist" aria-label="Choose sign-in type">
        <button type="button" role="tab" aria-selected={portal === "patient"} className={portal === "patient" ? "active" : ""} onClick={() => changePortal("patient")}>Patient</button>
        <button type="button" role="tab" aria-selected={portal === "admin"} className={portal === "admin" ? "active" : ""} onClick={() => changePortal("admin")}>Administrator</button>
      </div>

      {portal === "patient" ? <>
        <p className="access-intro">{IS_HOSTED_DEMO ? "Hosted public demo: do not enter real personal information. Use fictional details only. Age is calculated from the date of birth you enter." : "Enter your details to prepare a patient summary. Age is calculated from your date of birth."}</p>
        <form onSubmit={submitPatient}>
          <label className="patient-field"><span>Patient name</span><div><UserRound size={16} /><input autoComplete="name" maxLength={120} value={name} onChange={(event) => setName(event.target.value)} placeholder="Full name" required /></div></label>
          <label className="patient-field"><span>Date of birth</span><div><CalendarDays size={16} /><input type="date" max={localToday()} value={dateOfBirth} onChange={(event) => setDateOfBirth(event.target.value)} required /></div></label>
          {dateOfBirth && dateOfBirth <= localToday() && <p className="derived-age">Calculated age: <strong>{ageOnDate(dateOfBirth)} years</strong></p>}
          <label className="patient-field"><span>Email address</span><div><Mail size={16} /><input type="email" autoComplete="email" maxLength={254} value={email} onChange={(event) => setEmail(event.target.value)} placeholder="name@example.com" required /></div></label>
          <label className="patient-field"><span>Contact number</span><div><Phone size={16} /><input type="tel" autoComplete="tel" maxLength={25} value={phone} onChange={(event) => setPhone(event.target.value)} placeholder="Phone number" required /></div></label>
          {error && <p className="access-error" role="alert">{error}</p>}
          <button className="primary-button access-submit" type="submit" disabled={busy}>{busy ? "Opening patient session…" : <>Continue to MRI workbench <ArrowRight size={16} /></>}</button>
        </form>
        <div className="access-privacy"><LockKeyhole size={15} /><span>This prototype does not verify identity or create a patient account. {IS_HOSTED_DEMO ? "Use fictional details on this hosted demo. Details stay in this browser session and are sent to the hosted API only if you request a PDF summary." : "Details stay in this browser session and are included only in a summary you request."}</span></div>
      </> : <>
        <p className="access-intro">Administrator access includes the shared MRI workbench and the held-out model evaluation dashboard.</p>
        <form onSubmit={submitAdmin}>
          <label className="patient-field"><span>Username</span><div><UserRound size={16} /><input autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} required /></div></label>
          <label className="patient-field"><span>Password</span><div><LockKeyhole size={16} /><input type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} required /></div></label>
          {error && <p className="access-error" role="alert">{error}</p>}
          <button className="primary-button access-submit" type="submit" disabled={busy}>{busy ? "Signing in…" : "Sign in as administrator"} {!busy && <ArrowRight size={16} />}</button>
        </form>
        <div className="access-privacy"><ShieldCheck size={15} /><span>{IS_HOSTED_DEMO ? "Administrator credentials are configured in the hosting environment. The evaluation API requires an active admin session." : <>Local administrator credentials are configured in the project's ignored <b>.env</b> file. The evaluation API requires an active admin session.</>}</span></div>
      </>}
    </section>
    <footer className="patient-access-footer"><span>Experimental research software · Not for diagnosis or treatment</span><span>{IS_HOSTED_DEMO ? "Hosted research demo" : "Local workstation"}</span></footer>
  </main>;
}

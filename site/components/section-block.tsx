import { ReactNode } from "react";

interface SectionBlockProps {
  id?: string;
  eyebrow?: string;
  title: string;
  children: ReactNode;
};

export default function SectionBlock({
  id,
  eyebrow,
  title,
  children
}: SectionBlockProps) {
  return (
    <section id={id} className="section-block">
      {eyebrow ? <p className="section-eyebrow">{eyebrow}</p> : null}
      <h2 className="section-title">{title}</h2>
      <div className="section-body">{children}</div>
    </section>
  );
}

import { FigureRef } from "@/data/results";

interface FigureGridProps {
  title: string;
  figures: FigureRef[];
};

export default function FigureGrid({ title, figures }: FigureGridProps) {
  return (
    <section className="figure-section reveal">
      <h3>{title}</h3>
      <div className="figure-grid">
        {figures.map((figure) => (
          <article className="figure-card" key={figure.path}>
            <img src={figure.path} alt={figure.title} loading="lazy" />
            <h4>{figure.title}</h4>
            <p>{figure.caption}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

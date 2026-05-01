import { FigureRef } from "@/data/results";

interface FigureGridProps {
  title: string;
  figures: FigureRef[];
};

export default function FigureGrid({ title, figures }: FigureGridProps) {
  const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

  return (
    <section className="figure-section reveal">
      <h3>{title}</h3>
      <div className="figure-grid">
        {figures.map((figure) => {
          const resolvedPath = figure.path.startsWith("/")
            ? `${basePath}${figure.path}`
            : figure.path;

          return (
            <article className="figure-card" key={figure.path}>
              <img src={resolvedPath} alt={figure.title} loading="lazy" />
              <h4>{figure.title}</h4>
              <p>{figure.caption}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}

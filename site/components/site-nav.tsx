import Link from "next/link";

const NAV_ITEMS = [
  { href: "/", label: "Home" },
  { href: "/methods", label: "Methods" },
  { href: "/results", label: "Results" },
  { href: "/reproducibility", label: "Reproducibility" },
  { href: "/code", label: "Code" }
];

export default function SiteNav() {
  return (
    <nav aria-label="Primary">
      <ul className="nav-list">
        {NAV_ITEMS.map((item) => (
          <li key={item.href}>
            <Link href={item.href} className="nav-link">
              {item.label}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}

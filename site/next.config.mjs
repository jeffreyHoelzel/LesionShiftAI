const repoName = process.env.GITHUB_REPOSITORY?.split("/")[1] ?? "";
const defaultBasePath =
  process.env.GITHUB_ACTIONS === "true" && repoName ? `/${repoName}` : "";
const configuredBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? defaultBasePath;
const basePath =
  configuredBasePath.length > 1 && configuredBasePath.endsWith("/")
    ? configuredBasePath.slice(0, -1)
    : configuredBasePath;

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  basePath,
  assetPrefix: basePath,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath
  }
};

export default nextConfig;

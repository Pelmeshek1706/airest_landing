/** Resolve public media for both a domain root and a hosted subdirectory. */
export const publicAsset = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\//, '')}`;

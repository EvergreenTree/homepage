declare module 'glamor' {
  // Minimal typing to satisfy usage in this project
  // css(...) returns a className string
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export function css(...rules: any[]): string;
  // glamor v2 exposes `style`, which we alias to css in code
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export function style(...rules: any[]): string;
}

<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

## Architecture
- Keep the cardiac dashboard on the client-only index route so WebGL and browser interactions never render on the server.
- Keep illustrative clinical risk math in a pure, tested module shared by metrics, vessel mapping, and attribution views to ensure consistent updates.
- Use a custom procedural anatomical-style heart with schematic coronary curves because researched anatomical assets require attribution licenses outside the project's CC0 asset policy.
- Define visual colors in global semantic tokens and read those tokens for Three.js materials so the 3D scene and dashboard share one palette.

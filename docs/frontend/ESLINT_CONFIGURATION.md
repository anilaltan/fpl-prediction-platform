# ESLint Configuration Guide

## Overview

The project uses ESLint with TypeScript, React, and Next.js rules to enforce code quality and consistency across the frontend codebase.

## Configuration File

**Location**: `frontend/.eslintrc.json`

## Extends

- `next/core-web-vitals` - Next.js core web vitals rules
- `next/typescript` - Next.js TypeScript rules
- `plugin:@typescript-eslint/recommended` - TypeScript recommended rules
- `plugin:@typescript-eslint/recommended-requiring-type-checking` - TypeScript type-checking rules
- `plugin:react/recommended` - React recommended rules
- `plugin:react-hooks/recommended` - React Hooks rules

---

## Key Rules

### TypeScript Rules

- **`@typescript-eslint/no-unused-vars`**: Error on unused variables (allows `_` prefix)
- **`@typescript-eslint/no-explicit-any`**: Warn on `any` type usage
- **`@typescript-eslint/no-floating-promises`**: Error on unhandled promises
- **`@typescript-eslint/await-thenable`**: Error on awaiting non-promises
- **`@typescript-eslint/prefer-optional-chain`**: Prefer `?.` over `&&` chains
- **`@typescript-eslint/prefer-nullish-coalescing`**: Prefer `??` over `||` for null/undefined

### Modern ES6+ Rules

- **`prefer-const`**: Use `const` for variables that aren't reassigned
- **`no-var`**: Disallow `var` declarations
- **`prefer-arrow-callback`**: Prefer arrow functions for callbacks
- **`prefer-template`**: Prefer template literals over string concatenation
- **`prefer-destructuring`**: Prefer destructuring for objects
- **`prefer-spread`**: Prefer spread operator over `apply()`
- **`no-console`**: Warn on `console.log` (allows `console.warn` and `console.error`)

### React Rules

- **`react/react-in-jsx-scope`**: Off (Next.js handles this)
- **`react/prop-types`**: Off (using TypeScript for prop validation)
- **`react/jsx-key`**: Error on missing keys in lists
- **`react/jsx-no-duplicate-props`**: Error on duplicate props
- **`react/self-closing-comp`**: Error on non-self-closing components
- **`react/jsx-boolean-value`**: Error on `prop={true}` (use `prop` instead)
- **`react/jsx-curly-brace-presence`**: Error on unnecessary braces
- **`react/jsx-fragments`**: Prefer shorthand fragments (`<>...</>`)

### React Hooks Rules

- **`react-hooks/rules-of-hooks`**: Error on hooks rule violations
- **`react-hooks/exhaustive-deps`**: Warn on missing dependencies in hooks

### Code Quality Rules

- **`eqeqeq`**: Always use `===` and `!==` (except for null checks)
- **`curly`**: Always use curly braces for control statements
- **`no-else-return`**: Warn on unnecessary else after return
- **`no-nested-ternary`**: Warn on nested ternary operators
- **`object-shorthand`**: Prefer shorthand object properties
- **`prefer-object-spread`**: Prefer object spread over `Object.assign()`

### Next.js Rules

- **`@next/next/no-html-link-for-pages`**: Use Next.js `Link` component
- **`@next/next/no-img-element`**: Warn on `<img>` (use Next.js `Image`)
- **`@next/next/no-sync-scripts`**: Error on synchronous scripts

### Style Rules

- **`quotes`**: Single quotes (allows template literals)
- **`semi`**: No semicolons
- **`indent`**: 2 spaces
- **`max-len`**: 100 characters (warn, ignores URLs/strings/templates)
- **`max-lines`**: 500 lines per file (warn)
- **`max-lines-per-function`**: 100 lines per function (warn)
- **`comma-dangle`**: Always trailing commas in multiline

---

## File Overrides

### TypeScript Files

- `explicit-function-return-type`: Off (TypeScript infers return types)

### Next.js App Router Files

- `no-misused-promises`: Adjusted for async route handlers

### Test Files

- `@typescript-eslint/no-explicit-any`: Off
- `no-console`: Off

---

## Ignored Patterns

The following are ignored by ESLint:

- `.next/` - Next.js build output
- `node_modules/` - Dependencies
- `out/`, `dist/`, `build/` - Build outputs
- `*.config.js`, `*.config.ts` - Configuration files

See `frontend/.eslintignore` for complete list.

---

## Usage

### Run Linter

```bash
# From frontend directory
npm run lint

# Or with Docker
docker compose exec frontend npm run lint
```

### Fix Auto-fixable Issues

```bash
npm run lint -- --fix
```

### Check Specific Files

```bash
npm run lint -- app/layout.tsx components/MarketIntelligenceTable.tsx
```

---

## Integration with IDE

### VS Code / Cursor

Install the ESLint extension:
- Extension ID: `dbaeumer.vscode-eslint`

The extension will:
- Show linting errors inline
- Auto-fix on save (if configured)
- Display problems in the Problems panel

### Auto-fix on Save

Add to `.vscode/settings.json`:

```json
{
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "eslint.validate": [
    "javascript",
    "javascriptreact",
    "typescript",
    "typescriptreact"
  ]
}
```

---

## Common Issues & Solutions

### Issue: "Parsing error: Cannot find module '@typescript-eslint/parser'"

**Solution**: Install dependencies:
```bash
npm install --save-dev @typescript-eslint/parser @typescript-eslint/eslint-plugin
```

### Issue: "Type checking takes too long"

**Solution**: The `recommended-requiring-type-checking` rules require type checking. This is slower but more accurate. If too slow, you can remove it from `extends` array.

### Issue: "Unused variable" but it's used

**Solution**: Prefix with `_` to ignore:
```typescript
const _unusedVar = something() // No error
```

### Issue: "Promise must be handled"

**Solution**: Either await it or handle with `.catch()`:
```typescript
// ❌ Error
someAsyncFunction()

// ✅ Correct
await someAsyncFunction()

// ✅ Also correct
someAsyncFunction().catch(console.error)
```

---

## Best Practices

1. **Fix linting errors before committing**
   - Run `npm run lint` before pushing
   - Consider adding pre-commit hook

2. **Use TypeScript types instead of `any`**
   - If you must use `any`, add `// eslint-disable-next-line @typescript-eslint/no-explicit-any`

3. **Prefer modern ES6+ syntax**
   - Use `const`/`let` instead of `var`
   - Use arrow functions for callbacks
   - Use template literals for strings

4. **Follow React best practices**
   - Use functional components
   - Use hooks properly
   - Add keys to list items
   - Use self-closing tags when possible

5. **Keep functions small**
   - Aim for < 100 lines per function
   - Extract complex logic into separate functions

6. **Use meaningful variable names**
   - Avoid single-letter variables (except in loops)
   - Use descriptive names

---

## Customization

To customize rules, edit `frontend/.eslintrc.json`:

```json
{
  "rules": {
    "your-rule": "error" | "warn" | "off"
  }
}
```

### Disable Rule for Specific Line

```typescript
// eslint-disable-next-line rule-name
const problematicCode = something()
```

### Disable Rule for File

```typescript
/* eslint-disable rule-name */
```

---

## Related Documentation

- [TypeScript Configuration](../frontend/tsconfig.json)
- [Next.js ESLint Documentation](https://nextjs.org/docs/app/building-your-application/configuring/eslint)
- [TypeScript ESLint Rules](https://typescript-eslint.io/rules/)

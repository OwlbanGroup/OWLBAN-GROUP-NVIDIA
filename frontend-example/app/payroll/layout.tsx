import Link from 'next/link';
import { ReactNode } from 'react';

export default function PayrollLayout({ children }: { children: ReactNode }) {
  return (
    <div style={{ padding: 24 }}>
      <h1>Payroll</h1>
      <nav style={{ marginBottom: 16 }}>
        <Link href="/payroll">Dashboard</Link>{' '}
        | <Link href="/payroll/employees">Employees</Link>{' '}
        | <Link href="/payroll/runs">Runs</Link>
      </nav>
      <hr />
      <div style={{ marginTop: 16 }}>{children}</div>
    </div>
  );
}

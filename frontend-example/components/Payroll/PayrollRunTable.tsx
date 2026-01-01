import Link from 'next/link';

type PayrollRun = {
  id: string;
  runDate: string;
  periodStart: string;
  periodEnd: string;
  status: string;
  totalGross: string;
  totalNet: string;
};

type Props = {
  runs: PayrollRun[];
};

export function PayrollRunTable({ runs }: Props) {
  if (!runs.length) return <p>No payroll runs yet.</p>;

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          <th align="left">Run date</th>
          <th align="left">Period</th>
          <th align="right">Gross</th>
          <th align="right">Net</th>
          <th align="left">Status</th>
          <th align="left">Actions</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((r) => (
          <tr key={r.id}>
            <td>{new Date(r.runDate).toLocaleString()}</td>
            <td>
              {new Date(r.periodStart).toLocaleDateString()} –{' '}
              {new Date(r.periodEnd).toLocaleDateString()}
            </td>
            <td align="right">${Number(r.totalGross).toFixed(2)}</td>
            <td align="right">${Number(r.totalNet).toFixed(2)}</td>
            <td>{r.status}</td>
            <td>
              <Link href={`/payroll/runs/${r.id}`}>View / execute</Link>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

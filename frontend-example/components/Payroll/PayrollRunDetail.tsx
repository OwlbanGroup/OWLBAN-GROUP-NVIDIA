'use client';

import { useState } from 'react';
import { apiPost } from '../../lib/api';

type Payment = {
  id: string;
  grossPay: string;
  netPay: string;
  status: string;
  jpmPaymentId?: string;
  employee: {
    id: string;
    name: string;
    email: string;
  };
};

type Run = {
  id: string;
  runDate: string;
  periodStart: string;
  periodEnd: string;
  status: string;
  totalGross: string;
  totalNet: string;
  payments: Payment[];
};

type Props = {
  run: Run;
};

export function PayrollRunDetail({ run }: Props) {
  const [debitAccountId, setDebitAccountId] = useState('');
  const [executing, setExecuting] = useState(false);

  const canExecute = run.status === 'PENDING';

  async function handleExecute() {
    if (!debitAccountId) {
      alert('Enter debit account ID');
      return;
    }
    setExecuting(true);
    try {
      await apiPost(`/payroll/execute/${run.id}`, { debitAccountId });
      alert('Payroll run executed (backend will send payments).');
      window.location.reload();
    } catch (err) {
      console.error(err);
      alert('Failed to execute payroll run');
    } finally {
      setExecuting(false);
    }
  }

  return (
    <div>
      <h2>Payroll run</h2>
      <p>
        Run date: {new Date(run.runDate).toLocaleString()} <br />
        Period:{' '}
        {new Date(run.periodStart).toLocaleDateString()} –{' '}
        {new Date(run.periodEnd).toLocaleDateString()} <br />
        Status: {run.status} <br />
        Total gross: ${Number(run.totalGross).toFixed(2)} <br />
        Total net: ${Number(run.totalNet).toFixed(2)}
      </p>

      <h3>Payments</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th align="left">Employee</th>
            <th align="left">Email</th>
            <th align="right">Gross</th>
            <th align="right">Net</th>
            <th align="left">Status</th>
            <th align="left">JPM Payment ID</th>
          </tr>
        </thead>
        <tbody>
          {run.payments.map((p) => (
            <tr key={p.id}>
              <td>{p.employee.name}</td>
              <td>{p.employee.email}</td>
              <td align="right">${Number(p.grossPay).toFixed(2)}</td>
              <td align="right">${Number(p.netPay).toFixed(2)}</td>
              <td>{p.status}</td>
              <td>{p.jpmPaymentId || '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {canExecute && (
        <div style={{ marginTop: 24 }}>
          <h3>Execute payroll</h3>
          <p>Enter the bank account ID to debit for this payroll run.</p>
          <input
            placeholder="Debit account ID"
            value={debitAccountId}
            onChange={(e) => setDebitAccountId(e.target.value)}
          />
          <button onClick={handleExecute} disabled={executing}>
            {executing ? 'Executing...' : 'Execute payroll'}
          </button>
        </div>
      )}
    </div>
  );
}

'use client';

import { useState } from 'react';
import { apiPost } from '../../lib/api';

type Props = {
  orgId: string;
  onCreated?: () => void;
};

export function PayrollRunForm({ orgId, onCreated }: Props) {
  const [periodStart, setPeriodStart] = useState('');
  const [periodEnd, setPeriodEnd] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    try {
      await apiPost(`/payroll/run/${orgId}`, {
        periodStart: new Date(periodStart),
        periodEnd: new Date(periodEnd),
      });
      setPeriodStart('');
      setPeriodEnd('');
      onCreated?.();
    } catch (err) {
      console.error(err);
      alert('Failed to create payroll run');
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} style={{ marginBottom: 24 }}>
      <h3>Create payroll run</h3>
      <div>
        <label>
          Period start
          <input
            type="date"
            value={periodStart}
            onChange={(e) => setPeriodStart(e.target.value)}
            required
          />
        </label>
      </div>
      <div>
        <label>
          Period end
          <input
            type="date"
            value={periodEnd}
            onChange={(e) => setPeriodEnd(e.target.value)}
            required
          />
        </label>
      </div>
      <button type="submit" disabled={loading}>
        {loading ? 'Creating...' : 'Create run'}
      </button>
    </form>
  );
}

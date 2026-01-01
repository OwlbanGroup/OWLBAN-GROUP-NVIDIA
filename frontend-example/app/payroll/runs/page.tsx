import { apiGet } from '../../../lib/api';
import { PayrollRunForm } from '../../../components/Payroll/PayrollRunForm';
import { PayrollRunTable } from '../../../components/Payroll/PayrollRunTable';

const ORG_ID = 'ORG_UUID_HERE';

async function getRuns() {
  return apiGet<any[]>(`/payroll/runs/${ORG_ID}`);
}

export default async function PayrollRunsPage() {
  const runs = await getRuns();

  return (
    <div>
      <PayrollRunForm orgId={ORG_ID} onCreated={() => window.location.reload()} />
      <PayrollRunTable runs={runs} />
    </div>
  );
}

import { apiGet } from '../../../../lib/api';
import { PayrollRunDetail } from '../../../../components/Payroll/PayrollRunDetail';

type Props = {
  params: { runId: string };
};

async function getRun(runId: string) {
  return apiGet<any>(`/payroll/run/${runId}`);
}

export default async function RunDetailPage({ params }: Props) {
  const run = await getRun(params.runId);
  return <PayrollRunDetail run={run} />;
}

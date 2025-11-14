const faker = require('faker');
const moment = require('moment');

module.exports = {
  generateTelemetryData
};

function generateTelemetryData(requestParams, context, ee, next) {
  // Generate realistic telemetry data
  const telemetryData = {
    timestamp: moment().utc().format('YYYY-MM-DDTHH:mm:ss.SSS[Z]'),
    operation: faker.random.arrayElement(['CREATE', 'UPDATE', 'DELETE', 'READ', 'EXECUTE']),
    pfn: faker.random.alphaNumeric(16),
    version: faker.system.semver(),
    event_name: faker.random.arrayElement(['app_launch', 'user_action', 'system_event', 'error_occurred', 'performance_metric']),
    shell_id: faker.random.number({min: 1, max: 1000}),
    event_flags: faker.random.number({min: 0, max: 255}),
    pg_name: faker.internet.domainName(),
    dvc_sample: faker.random.float({min: 0, max: 1}),
    flags: faker.random.number({min: 0, max: 65535}),
    edition: faker.random.number({min: 1, max: 10}),
    epoch: moment().unix().toString(),
    seq: faker.random.number({min: 1, max: 1000000}),
    data_type: faker.random.number({min: 1, max: 100}),
    is_required: faker.random.boolean(),
    data_category: faker.random.number({min: 1, max: 50}),
    product: faker.random.number({min: 1, max: 100}),
    priv_tags: faker.random.number({min: 0, max: 4294967295}),
    policies: faker.random.number({min: 0, max: 4294967295}),
    cv: faker.random.alphaNumeric(8),
    boot_id: faker.random.number({min: 1, max: 1000000}),
    os_name: faker.random.arrayElement(['Windows', 'macOS', 'Linux', 'iOS', 'Android']),
    os_version: faker.system.semver(),
    exp_id: faker.random.alphaNumeric(12),
    app_id: faker.random.alphaNumeric(16),
    app_version: faker.system.semver(),
    is_1p: faker.random.number({min: 0, max: 1}),
    as_id: faker.random.number({min: 1, max: 1000}),
    local_id: faker.random.alphaNumeric(20),
    device_class: faker.random.arrayElement(['desktop', 'mobile', 'tablet', 'server']),
    dev_make: faker.company.companyName(),
    dev_model: faker.random.arrayElement(['Model A', 'Model B', 'Model C', 'Professional', 'Enterprise']),
    ticket_keys: JSON.stringify({
      ticket1: faker.random.alphaNumeric(32),
      ticket2: faker.random.alphaNumeric(32)
    }),
    user_local_id: faker.random.alphaNumeric(24),
    tz: faker.random.arrayElement(['UTC', 'EST', 'PST', 'GMT', 'CET']),
    pn1: faker.lorem.word(),
    p1: faker.random.alphaNumeric(10),
    pn2: faker.lorem.word(),
    p2: faker.random.alphaNumeric(10),
    pn3: faker.lorem.word(),
    p3: faker.random.alphaNumeric(10),
    pn4: faker.lorem.word(),
    p4: faker.random.alphaNumeric(10)
  };

  // Set the generated data in the context for use in the scenario
  context.vars.timestamp = telemetryData.timestamp;
  context.vars.operation = telemetryData.operation;
  context.vars.pfn = telemetryData.pfn;
  context.vars.version = telemetryData.version;
  context.vars.event_name = telemetryData.event_name;
  context.vars.shell_id = telemetryData.shell_id;
  context.vars.event_flags = telemetryData.event_flags;
  context.vars.pg_name = telemetryData.pg_name;
  context.vars.dvc_sample = telemetryData.dvc_sample;
  context.vars.flags = telemetryData.flags;
  context.vars.edition = telemetryData.edition;
  context.vars.epoch = telemetryData.epoch;
  context.vars.seq = telemetryData.seq;
  context.vars.data_type = telemetryData.data_type;
  context.vars.is_required = telemetryData.is_required;
  context.vars.data_category = telemetryData.data_category;
  context.vars.product = telemetryData.product;
  context.vars.priv_tags = telemetryData.priv_tags;
  context.vars.policies = telemetryData.policies;
  context.vars.cv = telemetryData.cv;
  context.vars.boot_id = telemetryData.boot_id;
  context.vars.os_name = telemetryData.os_name;
  context.vars.os_version = telemetryData.os_version;
  context.vars.exp_id = telemetryData.exp_id;
  context.vars.app_id = telemetryData.app_id;
  context.vars.app_version = telemetryData.app_version;
  context.vars.is_1p = telemetryData.is_1p;
  context.vars.as_id = telemetryData.as_id;
  context.vars.local_id = telemetryData.local_id;
  context.vars.device_class = telemetryData.device_class;
  context.vars.dev_make = telemetryData.dev_make;
  context.vars.dev_model = telemetryData.dev_model;
  context.vars.ticket_keys = telemetryData.ticket_keys;
  context.vars.user_local_id = telemetryData.user_local_id;
  context.vars.tz = telemetryData.tz;
  context.vars.pn1 = telemetryData.pn1;
  context.vars.p1 = telemetryData.p1;
  context.vars.pn2 = telemetryData.pn2;
  context.vars.p2 = telemetryData.p2;
  context.vars.pn3 = telemetryData.pn3;
  context.vars.p3 = telemetryData.p3;
  context.vars.pn4 = telemetryData.pn4;
  context.vars.p4 = telemetryData.p4;

  return next();
}
